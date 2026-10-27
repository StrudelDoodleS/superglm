"""The joint SCOP Newton step (Pya & Wood 2015, Appendix 1)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, Spline, SuperGLM
from superglm._roundoff import UNIT_ROUNDOFF as UNIT
from superglm.distributions import Tweedie
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.model.base import model_build_design_matrix
from superglm.solvers import irls_direct
from superglm.solvers.centered_system import build_centered_system
from superglm.solvers.rank import decompose_gram_if_authoritative
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import SCOPGroupSpec, SCOPJointNewton
from superglm.types import GroupSlice


def _curvature_solver_reparam(q_raw: int, kind: str):
    degree = 3
    n_interior = q_raw - degree - 1
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            np.linspace(0.0, 1.0, n_interior + 2)[1:-1] ** 1.7,
            np.ones(degree + 1),
        )
    )
    return build_scop_solver_reparam(
        q_raw, kind=kind, knots=knots, degree=degree, domain=(0.0, 1.0)
    )


def _spec(index, name, reparam, basis, bins=None):
    return SCOPGroupSpec(
        group_index=index,
        group=GroupSlice(name, 0, basis.shape[1]),
        reparam=reparam,
        B_scop=basis,
        S_scop=reparam.penalty_matrix(),
        bin_idx=bins,
    )


def _rows(spec):
    return spec.B_scop if spec.bin_idx is None else spec.B_scop[spec.bin_idx]


def _working_problem(*, residual_shift=0.0, lambdas=None, seed=7):
    """A penalized Gaussian working model: intercept, 3 ordinary columns, two SCOP groups.

    The ordinary columns share most of their variation with the dense SCOP
    group, so the two blocks are strongly coupled.
    """
    rng = np.random.default_rng(seed)
    n, n_bins = 400, 30
    convex = _curvature_solver_reparam(6, "convex")
    increasing = build_scop_solver_reparam(5, kind="increasing")
    dense = rng.normal(size=(n, convex.q))
    bins = rng.integers(0, n_bins, size=n)
    specs = (
        _spec(1, "a", convex, dense),
        _spec(2, "b", increasing, rng.normal(size=(n_bins, increasing.q)), bins),
    )
    X = 0.8 * dense[:, :3] + 0.6 * rng.normal(size=(n, 3))
    thetas = [rng.normal(scale=0.3, size=convex.q), rng.normal(scale=0.3, size=increasing.q)]
    b = rng.normal(size=3)
    alpha = 0.4
    weights = rng.uniform(0.3, 2.0, size=n)
    eta = alpha + X @ b + sum(_rows(s) @ s.reparam.forward(t) for s, t in zip(specs, thetas))
    z = eta + rng.normal(scale=0.3, size=n) + residual_shift
    return dict(
        X=X,
        S_r=0.5 * np.eye(3),
        specs=specs,
        lambdas={"a": 0.7, "b": 0.3} if lambdas is None else lambdas,
        thetas=thetas,
        b=b,
        alpha=alpha,
        W=weights,
        z=z,
    )


def _newton_oracle(problem, *, gauss_newton=False):
    """``-H^-1 g`` of the working objective in (intercept, b, theta), formed densely."""
    specs, thetas, W = problem["specs"], problem["thetas"], problem["W"]
    rows = [_rows(s) for s in specs]
    mapped = [s.reparam.forward(t) for s, t in zip(specs, thetas)]
    jacobians = [s.reparam.jacobian_diagonal(t) for s, t in zip(specs, thetas)]
    seconds = [s.reparam.second_derivative_diagonal(t) for s, t in zip(specs, thetas)]
    X = problem["X"]
    n, p = X.shape
    residual = problem["z"] - (
        problem["alpha"] + X @ problem["b"] + sum(r @ g for r, g in zip(rows, mapped))
    )
    Z = np.hstack([np.ones((n, 1)), X, *[r * j for r, j in zip(rows, jacobians)]])
    width = Z.shape[1]
    penalty = np.zeros((width, width))
    penalty[1 : 1 + p, 1 : 1 + p] = problem["S_r"]
    start = 1 + p
    for spec in specs:
        q = spec.S_scop.shape[0]
        penalty[start : start + q, start : start + q] = (
            problem["lambdas"][spec.group.name] * spec.S_scop
        )
        start += q
    point = np.concatenate([[problem["alpha"]], problem["b"], *thetas])
    hessian = Z.T @ (W[:, None] * Z) + penalty
    gradient = -Z.T @ (W * residual) + penalty @ point
    if not gauss_newton:
        curvature = -np.concatenate(seconds) * np.concatenate([r.T @ (W * residual) for r in rows])
        hessian[1 + p :, 1 + p :] += np.diag(curvature)
    return -np.linalg.solve(hessian, gradient), hessian


def _joint_step(problem):
    """The library's step: the ordinary solve with theta fixed, then ``SCOPJointNewton``."""
    X, W, z = problem["X"], problem["W"], problem["z"]
    n, p = X.shape
    design = DesignMatrix([DenseGroupMatrix(X)], n=n, p=p)
    engine = SCOPJointNewton(problem["specs"], problem["lambdas"], design, n)
    scop_eta = engine.predictor(problem["thetas"])
    reduced = build_centered_system(dm=design, W=W, z_off=z - scop_eta, penalty=problem["S_r"])
    rank = decompose_gram_if_authoritative(reduced.hessian)
    b_fixed = rank.solve(reduced.rhs)
    step = engine.step(
        problem["thetas"],
        W=W,
        z=z,
        eta=problem["alpha"] + X @ problem["b"] + scop_eta,
        reduced=reduced,
        solve=rank.solve,
        ordinary_change=b_fixed - problem["b"],
    )
    b_new = b_fixed + step.ordinary_correction
    alpha_new = reduced.mean_z - step.intercept_shift - float(reduced.mean_x @ b_new)
    theta_change = np.concatenate(
        [
            step.results[s.group_index].beta_new - t
            for s, t in zip(problem["specs"], problem["thetas"])
        ]
    )
    change = np.concatenate([[alpha_new - problem["alpha"]], b_new - problem["b"], theta_change])
    return change, step


def _relative_error(actual, expected):
    return float(np.max(np.abs(actual - expected)) / max(1.0, float(np.max(np.abs(expected)))))


def test_joint_step_is_the_newton_step_on_every_coefficient():
    problem = _working_problem()
    expected, hessian = _newton_oracle(problem)
    change, step = _joint_step(problem)

    assert not any(result.used_fisher_fallback for result in step.results.values())
    # A backward-stable solve of either form is accurate to its condition number.
    bound = 64 * hessian.shape[0] * np.linalg.cond(hessian) * UNIT
    assert _relative_error(change, expected) <= bound
    # The block-coordinate step the joint step replaces (the ordinary block
    # held at its fixed-theta solve) is not that step on this coupled fixture.
    uncoupled = change.copy()
    uncoupled[1:4] -= step.ordinary_correction
    assert _relative_error(uncoupled, expected) > 1e3 * bound


def test_indefinite_map_curvature_takes_the_gauss_newton_block():
    problem = _working_problem(residual_shift=1_000.0)
    expected, hessian = _newton_oracle(problem, gauss_newton=True)
    change, step = _joint_step(problem)

    assert all(result.used_fisher_fallback for result in step.results.values())
    bound = 64 * hessian.shape[0] * np.linalg.cond(hessian) * UNIT
    assert _relative_error(change, expected) <= bound
    # The step solved with the Schur block; the curvature it reports is the
    # block's own Gauss-Newton Hessian J G J + lambda S, without the map's
    # curvature E, on the range the step resolved (all of it here).
    start = 1 + problem["X"].shape[1]
    for spec in problem["specs"]:
        result = step.results[spec.group_index]
        width = spec.S_scop.shape[0]
        own = hessian[start : start + width, start : start + width]
        assert result.discarded_directions.shape[0] == 0
        assert np.max(np.abs(result.H_penalized - own)) <= 64 * len(problem["W"]) * UNIT * np.max(
            np.abs(own)
        )
        start += width


def test_a_coordinate_at_its_log_space_boundary_is_discarded_and_held():
    problem = _working_problem(lambdas={"a": 0.7, "b": 0.0})
    problem["thetas"][1][2] = -40.0
    _, step = _joint_step(problem)

    held = step.results[2]
    assert held.discarded_directions.shape == (1, 4)
    assert step.results[1].discarded_directions.shape == (1, 5)
    assert abs(held.discarded_directions[0, 2]) > 1.0 - 1e-8
    theta_change = held.beta_new - problem["thetas"][1]
    scale = max(1.0, float(np.max(np.abs(theta_change))))
    assert abs(theta_change[2]) <= 1e-10 * scale


def test_a_non_finite_system_takes_no_step_and_warns_nothing():
    problem = _working_problem()
    problem["thetas"] = [np.full(5, 600.0), np.full(4, 600.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, step = _joint_step(problem)

    for spec, theta in zip(problem["specs"], problem["thetas"]):
        result = step.results[spec.group_index]
        np.testing.assert_array_equal(result.beta_new, theta)
        assert result.step_norm == 0.0
    np.testing.assert_array_equal(step.ordinary_correction, np.zeros(3))
    assert step.intercept_shift == 0.0


def test_block_curvature_is_the_finite_difference_hessian():
    """The retained slope has identity, not exponential, map curvature."""
    rng = np.random.default_rng(216)
    reparam = _curvature_solver_reparam(8, "convex")
    q = reparam.q
    n = 100
    basis = rng.standard_normal((n, q))
    W = rng.uniform(0.2, 2.0, size=n)
    theta = rng.normal(scale=0.3, size=q)
    lam = 0.4
    mapped = reparam.forward(theta)
    z = basis @ mapped + 0.02 * rng.standard_normal(n)
    engine = SCOPJointNewton(
        [_spec(0, "x", reparam, basis)], {"x": lam}, DesignMatrix([], n=n, p=0), n
    )

    result = engine.block_curvature([theta], W=W, z=z, eta=basis @ mapped)[0]

    def gradient(parameters):
        residual = z - basis @ reparam.forward(parameters)
        return -reparam.jacobian_diagonal(parameters) * (basis.T @ (W * residual)) + lam * (
            reparam.penalty_matrix() @ parameters
        )

    step = 2e-5
    numerical = np.column_stack(
        [(gradient(theta + step * e) - gradient(theta - step * e)) / (2 * step) for e in np.eye(q)]
    )
    assert abs((basis.T @ (W * (z - basis @ mapped)))[0]) > 0.1
    assert result.used_fisher_fallback is False
    assert result.discarded_directions.shape == (0, q)
    np.testing.assert_allclose(result.H_penalized, numerical, rtol=2e-9, atol=3e-7)
    # an exp-style second derivative on the free slope would add the score here
    weighted_gram = basis.T @ (W[:, None] * basis)
    np.testing.assert_allclose(result.H_penalized[0, 0], weighted_gram[0, 0], rtol=2e-14)


def _correlated_monotone_frame(corr, n=5000, seed=0):
    """Poisson rates with a smooth in x1 and an increasing step in x2 that follows x1."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = np.clip(corr * x1 + (1.0 - corr) * rng.uniform(0.0, 1.0, n), 0.0, 1.0)
    exposure = rng.uniform(0.1, 1.0, n)
    rate = np.exp(-1.0 + 0.6 * np.sin(2 * np.pi * x1) + 1.5 / (1.0 + np.exp(-10.0 * (x2 - 0.5))))
    y = rng.poisson(exposure * rate).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2}), y, np.log(exposure)


def _monotone_model():
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=False,
        features={
            "x1": Spline(kind="ps", k=12),
            "x2": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
        },
    )


def test_strongly_coupled_monotone_fit_converges():
    """corr(x1, x2) = 0.9: the block-coordinate inner fit contracted at 0.99 and never certified."""
    frame, y, offset = _correlated_monotone_frame(0.9)
    model = _monotone_model()
    model.fit_reml(frame, y, offset=offset)

    diagnostics = model.reml_diagnostics()
    assert diagnostics["converged"]
    assert diagnostics["n_reml_iter"] > 0
    assert model.result.converged


def test_inner_fit_at_fixed_lambda_needs_few_iterations():
    """A cold inner fit on a coupled fixture at fixed lambda: a count, never a time.

    At corr(x1, x2) = 0.7 the block-coordinate iteration contracted at 0.87 per
    iteration and did not meet its step test in 100; the joint Newton step
    converges quadratically, in 10 here.
    """
    frame, y, offset = _correlated_monotone_frame(0.7)
    model = _monotone_model()
    y_fit, weights, offset_fit = model_build_design_matrix(model, frame, y, np.ones_like(y), offset)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y_fit,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x1": 8.94, "x2": 0.770},
        offset=offset_fit,
        max_iter=100,
        tol=1e-6,
        convergence="coefficients",
        weight_semantics="frequency",
    )

    assert result.converged
    assert result.n_iter <= 20


def _fit_tweedie_scop(n=4000, profile=None):
    """A Tweedie/log fit of a smooth and an increasing SCOP spline, heavy-tailed pure premiums."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
    exposure = rng.uniform(0.02, 1.0, n)
    rate = 0.1 * np.exp(
        0.5 * np.sin(2 * np.pi * frame["x1"]) + 1.2 / (1 + np.exp(-8 * (frame["x2"] - 0.5)))
    )
    counts = rng.poisson(rate * exposure)
    y = np.array([rng.gamma(0.5, 2000.0, c).sum() for c in counts]) / exposure
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        discrete=False,
        features={
            "x1": Spline(kind="ps", k=10),
            "x2": Spline(kind="ps", k=8, constraint=Constraint.fit.increasing),
        },
    )
    y_fit, weights, offset = model_build_design_matrix(model, frame, y, exposure, None)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y_fit,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x1": 1.0, "x2": 1.0},
        offset=offset,
        max_iter=100,
        tol=1e-10,
        convergence="coefficients",
        weight_semantics="prior",
        profile=profile,
    )
    return result, model, y_fit, weights


def test_tweedie_log_scop_fit_takes_observed_newton_rows():
    """The joint step is full Newton: Tweedie/log rows carry the observed information.

    On Fisher rows a Tweedie/log SCOP inner fit contracts only linearly where
    the observed information exceeds Fisher's, as heavy-tailed pure premiums
    give: 0.8 to 0.94 per iteration on 50,000 freMTPL2 rows, where most inner
    fits exhausted their 100 iterations.
    """
    profile: dict = {}
    result, *_ = _fit_tweedie_scop(profile=profile)

    assert result.converged
    assert profile.get("irls_observed_newton_iters", 0) == result.n_iter


def test_the_scop_line_search_rejects_trials_without_finite_observed_rows(monkeypatch):
    """An observed fit's SCOP trials pass the finiteness test its ordinary trials do.

    Every accepted trial then has finite observed rows, so the next
    iteration's rows can only be non-finite at the entry state.
    """
    from types import SimpleNamespace

    predicates = []
    select = irls_direct._select_irls_trial

    def recorded(**kwargs):
        predicates.append(kwargs.get("invalid_state"))
        return select(**kwargs)

    monkeypatch.setattr(irls_direct, "_select_irls_trial", recorded)
    result, model, y, weights = _fit_tweedie_scop(n=1000)

    assert result.converged
    assert predicates and all(predicate is not None for predicate in predicates)
    eta = np.zeros_like(y)
    assert not predicates[0](SimpleNamespace(eta=eta, mu=np.exp(eta)))
    eta[0] = 800.0  # past exp's range: the observed row is not finite
    with np.errstate(over="ignore"):
        assert predicates[0](SimpleNamespace(eta=eta, mu=np.exp(eta)))


def test_a_rejected_last_step_publishes_the_retained_models_curvature(monkeypatch):
    """The terminal SCOP curvature is the block's own at the retained theta, however the loop ends.

    A rejected last step used to leave the curvature, fallback flag and
    discarded directions of the step before it, taken at the previous
    iterate on the step's Schur block.
    """
    from superglm.solvers.irls_state import _IRLSStepDecision

    selections = []
    select = irls_direct._select_irls_trial

    def reject_third(**kwargs):
        selections.append(kwargs)
        if len(selections) == 3:
            return _IRLSStepDecision(alpha=0.0, step_halvings=0, step_rejected=True)
        return select(**kwargs)

    terminal = []
    block_curvature = SCOPJointNewton.block_curvature

    def recorded(self, thetas, **kwargs):
        curvature = block_curvature(self, thetas, **kwargs)
        terminal.append(([np.array(theta) for theta in thetas], curvature))
        return curvature

    monkeypatch.setattr(irls_direct, "_select_irls_trial", reject_third)
    monkeypatch.setattr(SCOPJointNewton, "block_curvature", recorded)
    frame, y, offset = _correlated_monotone_frame(0.7)
    model = _monotone_model()
    y_fit, weights, offset_fit = model_build_design_matrix(model, frame, y, np.ones_like(y), offset)
    result, _, states = irls_direct.fit_irls_direct(
        model._dm,
        y_fit,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x1": 8.94, "x2": 0.770},
        offset=offset_fit,
        max_iter=100,
        tol=1e-6,
        convergence="coefficients",
        weight_semantics="frequency",
        return_scop_state=True,
    )

    assert not result.converged
    assert len(selections) == 3
    ((thetas, curvature),) = terminal
    for theta, gi in zip(thetas, sorted(states), strict=True):
        np.testing.assert_array_equal(theta, states[gi]["beta_eff"])
        np.testing.assert_array_equal(states[gi]["H_scop_penalized"], curvature[gi].H_penalized)
        assert states[gi]["last_fisher_fallback"] == curvature[gi].used_fisher_fallback


def test_cross_block_routes_form_the_centred_cross_product():
    """Both routes for bounded groups form X~_r' W X~_s from centred columns, to their rounding."""
    from superglm import Categorical
    from superglm.solvers.scop_newton import _JointWeightCross

    rng = np.random.default_rng(11)
    n = 5000
    frame = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, n),
            "b": rng.uniform(0, 1, n),
            "m": rng.uniform(0, 1, n),
            "c": rng.choice(list("pqrst"), n, p=[0.9, 0.04, 0.03, 0.02, 0.01]),
        }
    )
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "a": Spline(kind="ps", k=8),
            "b": Spline(kind="ps", k=6),
            "m": Spline(kind="ps", k=8, constraint=Constraint.fit.increasing),
            "c": Categorical(base="most_exposed"),
        },
        interactions=[("a", "b")],
    )
    model_build_design_matrix(model, frame, rng.poisson(1.0, n).astype(float), np.ones(n), None)
    pairs = list(zip(model._groups, model._dm.group_matrices))
    scop_group, scop = next((g, gm) for g, gm in pairs if g.monotone_engine == "scop")
    ordinary = [gm for g, gm in pairs if g.monotone_engine != "scop"]
    reduced = DesignMatrix(ordinary, n=n, p=sum(gm.shape[1] for gm in ordinary))
    reparam = build_scop_solver_reparam(scop.B_scop_unique.shape[1] + 1, kind="increasing")
    spec = SCOPGroupSpec(
        0, scop_group, reparam, scop.B_scop_unique, reparam.penalty_matrix(), scop.bin_idx
    )
    engine = SCOPJointNewton([spec], {scop_group.name: 1.0}, reduced, n)
    W = rng.uniform(0.1, 3.0, n)
    z = rng.normal(size=n)
    moments = engine._moments([np.zeros(reparam.q)], W=W, z=z, eta=np.zeros(n))
    system = build_centered_system(dm=reduced, W=W, z_off=z, penalty=np.zeros((reduced.p,) * 2))

    X = reduced.toarray()
    S = scop.B_scop_unique[scop.bin_idx]
    centred_X = X - (W @ X) / W.sum()
    centred_S = S - (W @ S) / W.sum()
    oracle = centred_X.T @ (W[:, None] * centred_S)
    # every route sums n weighted products, then at most n support or bin rows,
    # of operands centred to within their means' rounding (Higham 2002, eq. 3.5)
    gamma = 4 * (2 * n + 2) * UNIT
    assert isinstance(engine._cross.tables, _JointWeightCross)
    one_pass, _ = engine._cross.moment(W, moments, system)
    assert np.all(
        np.abs(one_pass - oracle)
        <= gamma * (np.abs(centred_X).T @ (W[:, None] * np.abs(centred_S)))
    )
    # the per-pair kernels take the bounded rows uncentred against the centred SCOP support
    engine._cross.tables = None
    per_pair, norms = engine._cross.moment(W, moments, system)
    assert np.all(
        np.abs(per_pair - oracle) <= gamma * (np.abs(X).T @ (W[:, None] * np.abs(centred_S)))
    )
    np.testing.assert_allclose(norms, np.sqrt(np.sum(W[:, None] * X**2, axis=0)), rtol=n * UNIT)
    # a group with no compact support (a dense block) keeps the per-pair plan
    dense = DesignMatrix(
        [*ordinary, DenseGroupMatrix(frame[["a"]].to_numpy())], n=n, p=reduced.p + 1
    )
    assert _JointWeightCross.build(dense, [(scop.B_scop_unique, scop.bin_idx)]) is None


def _fit_translated(shift):
    """The joint step iterated on an exact Gaussian model with one ordinary column at ``shift``."""
    s = np.array([-0.4375, -0.1875, 0.0625, 0.5625])[:, None]
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    W = np.array([1.0, 2.0, 4.0, 8.0])
    y = 0.2 + 0.3 * x + 2.0 * s[:, 0]
    X = (x + shift)[:, None]
    design = DesignMatrix([DenseGroupMatrix(X)], n=4, p=1)
    reparam = build_scop_solver_reparam(2, kind="increasing")
    engine = SCOPJointNewton([_spec(0, "s", reparam, s)], {"s": 0.0}, design, 4)
    theta, b, eta = np.zeros(1), np.zeros(1), engine.predictor([np.zeros(1)])
    for _ in range(30):
        reduced = build_centered_system(
            dm=design, W=W, z_off=y - engine.predictor([theta]), penalty=np.zeros((1, 1))
        )
        rank = decompose_gram_if_authoritative(reduced.hessian)
        b_fixed = rank.solve(reduced.rhs)
        step = engine.step(
            [theta],
            W=W,
            z=y,
            eta=eta,
            reduced=reduced,
            solve=rank.solve,
            ordinary_change=b_fixed - b,
        )
        b = b_fixed + step.ordinary_correction
        theta = step.results[0].beta_new
        centre, centre_lo = reduced.centre_pair()
        rows = (X - centre) - (0.0 if centre_lo is None else centre_lo)
        level = reduced.mean_z - step.intercept_shift
        eta = level + rows @ b + engine.predictor([theta])
    return float(reparam.forward(theta)[0]), float(b[0]), float(W @ (y - eta) ** 2)


def test_a_translated_ordinary_column_leaves_the_joint_step_unchanged():
    """The cross block is a product of centred columns: an exact 1.5e15 shift of x changes nothing.

    From raw moments the shift turned the centred cross product 4.7 into 5.0,
    and the fit stopped at the SCOP coefficient 1 with deviance 0.067.
    """
    for shift in (0.0, 1.5e15):
        gamma, slope, deviance = _fit_translated(shift)
        # an exact model on a well-conditioned 3-parameter system
        assert abs(gamma - 2.0) <= 64 * UNIT * 2.0
        assert abs(slope - 0.3) <= 64 * UNIT
        assert deviance <= (64 * UNIT) ** 2 * 15.0


def test_a_translated_ordinary_column_leaves_a_damped_step_unchanged(monkeypatch):
    """x on [-8, 8] beside a convex spline of another column, weights exp(U(-3, 3)), no penalty (Sol's repro).

    The line search damps six of the fit's steps. A damped trial's ordinary
    level was read back from the raw intercept, and its predictor formed
    from that intercept and the raw columns, both at u |m' b|, about 0.05
    once x is translated by 1.5e15: every trial was refused at iteration 6
    and the fit stopped there at deviance 2.33 (1.22 untranslated). Each
    trial now forms its predictor about the system's centre, as its
    proposal does, from the committed state's own level.
    """
    from superglm import Numeric, PSpline

    steps = []
    select = irls_direct._select_irls_trial

    def recorded(**kwargs):
        decision = select(**kwargs)
        steps.append((decision.step_halvings, decision.step_rejected))
        return decision

    monkeypatch.setattr(irls_direct, "_select_irls_trial", recorded)
    rng = np.random.default_rng(5)
    x = rng.integers(-8, 9, 200).astype(float)
    s = rng.uniform(size=200)
    y = 0.2 + 0.3 * x + 2.2 * s + 0.6 * s * s + 0.05 * rng.normal(size=200)
    weights = np.exp(rng.uniform(-3, 3, 200))
    fits = {}
    for shift in (0.0, 1.5e15):
        steps.clear()
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={"x": Numeric(), "s": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
        )
        model.fit(pd.DataFrame({"x": x + shift, "s": s}), y, sample_weight=weights)
        fits[shift] = (model.result, list(steps))

    (plain, plain_steps), (shifted, shifted_steps) = fits.values()
    assert sum(halvings > 0 for halvings, _ in plain_steps) >= 2
    assert shifted_steps == plain_steps
    assert plain.converged and shifted.converged
    assert abs(shifted.deviance - plain.deviance) <= model._tol * plain.deviance


def _repeated_rows_spec(index, rng, n, distinct, reparam):
    """An exact SCOP group whose ``n`` rows take ``distinct`` distinct values."""
    values = rng.normal(size=(distinct, reparam.q))
    codes = np.concatenate([np.arange(distinct), rng.integers(0, distinct, n - distinct)])
    return _spec(index, f"g{index}", reparam, values[rng.permutation(codes)])


def test_exact_supports_compress_only_when_they_pay_and_pair_tables_are_capped(monkeypatch):
    """No SCOP pair allocates a weight table past the cap: a count of cells, never a time."""
    from superglm.solvers import scop_newton

    cells = []
    table = scop_newton._disc_disc_2d_hist

    def counted(bins_i, bins_j, W, rows_i, rows_j):
        cells.append(rows_i * rows_j)
        return table(bins_i, bins_j, W, rows_i, rows_j)

    monkeypatch.setattr(scop_newton, "_disc_disc_2d_hist", counted)
    rng = np.random.default_rng(4)
    increasing = build_scop_solver_reparam(6, kind="increasing")

    # Almost-unique supports (n - 1 distinct rows) stay at row level.
    n = 3000
    specs = [_repeated_rows_spec(k, rng, n, n - 1, increasing) for k in (0, 1)]
    engine = SCOPJointNewton(specs, {"g0": 1.0, "g1": 1.0}, DesignMatrix([], n=n, p=0), n)
    assert all(bins is None for _, bins in engine.columns)
    thetas = [np.zeros(increasing.q), np.zeros(increasing.q)]
    W = rng.uniform(0.5, 2.0, n)
    engine.block_curvature(thetas, W=W, z=rng.normal(size=n), eta=np.zeros(n))
    assert cells == []

    # Supports at most half the rows compress; their pair, past the cap, takes no table.
    n = 5000
    distinct = 2300
    assert distinct**2 > scop_newton._MAX_DISC_DISC_HIST_CELLS
    specs = [_repeated_rows_spec(k, rng, n, distinct, increasing) for k in (0, 1)]
    engine = SCOPJointNewton(specs, {"g0": 1.0, "g1": 1.0}, DesignMatrix([], n=n, p=0), n)
    assert [basis.shape[0] for basis, _ in engine.columns] == [distinct, distinct]
    W = rng.uniform(0.5, 2.0, n)
    engine.block_curvature(thetas, W=W, z=rng.normal(size=n), eta=np.zeros(n))
    assert max(cells, default=0) <= scop_newton._MAX_DISC_DISC_HIST_CELLS
    cross = scop_newton._compute_cross_gram(*engine.columns[0], *engine.columns[1], W)
    rows = [spec.B_scop for spec in specs]
    # sums of the same n products in another order (Higham 2002, eq. 3.5)
    bound = 2 * (2 * n + 2) * UNIT * (np.abs(rows[0]).T @ (W[:, None] * np.abs(rows[1])))
    assert np.all(np.abs(cross - rows[0].T @ (W[:, None] * rows[1])) <= bound)


def test_a_categorical_cross_block_never_forms_its_identity(monkeypatch):
    """A K-level categorical's (K + 1, K) identity support is read off its table, never allocated."""
    from superglm import Categorical
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        CrossMatrixExecutionPlan,
        DiscretizedSCOPGroupMatrix,
        MatrixExecutionPlan,
    )
    from superglm.solvers import scop_newton

    formed = []
    support = scop_newton._compact_support
    monkeypatch.setattr(
        scop_newton, "_compact_support", lambda gm: (formed.append(type(gm)), support(gm))[1]
    )
    rng = np.random.default_rng(12)
    n, levels = 6000, 1500
    frame = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, n),
            "m": rng.uniform(0, 1, n),
            "c": np.concatenate([np.arange(levels), rng.integers(0, levels, n - levels)]).astype(
                str
            ),
        }
    )
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "a": Spline(kind="ps", k=8),
            "m": Spline(kind="ps", k=8, constraint=Constraint.fit.increasing),
            "c": Categorical(base="most_exposed"),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_build_design_matrix(model, frame, rng.poisson(1.0, n).astype(float), np.ones(n), None)
    pairs = list(zip(model._groups, model._dm.group_matrices))
    scop = next(gm for g, gm in pairs if g.monotone_engine == "scop")
    ordinary = [gm for g, gm in pairs if g.monotone_engine != "scop"]
    columns = [(scop.B_scop_unique, scop.bin_idx)]
    categorical = next(i for i, gm in enumerate(ordinary) if isinstance(gm, CategoricalGroupMatrix))
    reduced = DesignMatrix(ordinary, n=n, p=sum(gm.shape[1] for gm in ordinary))

    one_pass = scop_newton._JointWeightCross.build(reduced, columns)
    assert CategoricalGroupMatrix not in formed
    assert one_pass.layout.supports[categorical] == (None, None)
    weights = rng.uniform(0.1, 3.0, n)
    sum_w = float(weights.sum())
    mass = np.bincount(scop.bin_idx, weights, scop.B_scop_unique.shape[0])
    centred, _ = scop_newton._centre_rows(scop.B_scop_unique, mass, sum_w)
    plan = CrossMatrixExecutionPlan(
        reduced.execution_plan,
        MatrixExecutionPlan([DiscretizedSCOPGroupMatrix(centred, scop.bin_idx)], n=n),
    )
    mean_x = reduced.rmatvec(weights) / sum_w
    expected = plan.cross_moment(weights) - np.outer(mean_x, centred.T @ mass)
    # sums of the same n products in another order: within n u of the largest entry
    np.testing.assert_allclose(
        one_pass.cross_moment(weights, [(centred, scop.bin_idx)], sum_w),
        expected,
        rtol=0,
        atol=n * UNIT * np.abs(expected).max(),
    )

    # Tables past the cap are refused from the groups' metadata, before any support is formed.
    formed.clear()
    monkeypatch.setattr(scop_newton, "_MAX_PACKED_HIST_CELLS", levels * scop.B_scop_unique.shape[0])
    fresh = DesignMatrix(ordinary, n=n, p=reduced.p)
    assert scop_newton._JointWeightCross.build(fresh, columns) is None
    assert formed == []


def test_packed_codes_past_their_byte_budget_are_refused_before_any_support(monkeypatch):
    """The one-pass cross block's packed codes, 4 bytes per row and group, are sized from the design first.

    At the budget the layout is built; one byte under it the route is
    refused, and no group's compact support is formed.
    """
    from superglm import Categorical
    from superglm.solvers import scop_newton

    formed = []
    support = scop_newton._compact_support
    monkeypatch.setattr(
        scop_newton, "_compact_support", lambda gm: (formed.append(type(gm)), support(gm))[1]
    )
    rng = np.random.default_rng(13)
    n = 500
    frame = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, n),
            "m": rng.uniform(0, 1, n),
            "c": rng.integers(0, 5, n).astype(str),
        }
    )
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "a": Spline(kind="ps", k=8),
            "m": Spline(kind="ps", k=8, constraint=Constraint.fit.increasing),
            "c": Categorical(base="most_exposed"),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_build_design_matrix(model, frame, rng.poisson(1.0, n).astype(float), np.ones(n), None)
    pairs = list(zip(model._groups, model._dm.group_matrices))
    scop = next(gm for g, gm in pairs if g.monotone_engine == "scop")
    ordinary = [gm for g, gm in pairs if g.monotone_engine != "scop"]
    columns = [(scop.B_scop_unique, scop.bin_idx)]
    width = sum(gm.shape[1] for gm in ordinary)
    packed = 4 * n * len(ordinary)

    monkeypatch.setattr(scop_newton, "_MAX_CROSS_EXPANSION_BYTES", packed - 1)
    refused = scop_newton._JointWeightCross.build(DesignMatrix(ordinary, n=n, p=width), columns)
    assert refused is None
    assert formed == []
    monkeypatch.setattr(scop_newton, "_MAX_CROSS_EXPANSION_BYTES", packed)
    built = scop_newton._JointWeightCross.build(DesignMatrix(ordinary, n=n, p=width), columns)
    assert built is not None
    assert built.layout.codes.nbytes == packed


def test_rounding_on_a_null_direction_is_dropped_whatever_its_sign():
    """A Gram that cannot decide hands the decision to the rows, which drop an exact null.

    The Gram's null direction carries half its formation bound, of either
    sign; the rows are null there and resolve every other direction.
    """
    from superglm.solvers.rank import _gamma_up
    from superglm.solvers.scop_newton import _resolved_root

    rng = np.random.default_rng(3)
    q, n = 5, 10_000
    count = 2 * n + 2
    basis, _ = np.linalg.qr(rng.normal(size=(q, q)))
    null = basis[:, 0]
    scale = rng.uniform(0.5, 4.0, q)
    values = np.array([0.0, 1e-6, 0.1, 1.0, 3.0])
    left, _ = np.linalg.qr(rng.normal(size=(n, q)))
    rows = (left * np.sqrt(values)) @ basis.T
    factor = np.linalg.qr(rows * scale[None, :], mode="r")
    for rounding in (-0.5, 0.5):
        values[0] = rounding * _gamma_up(count) * q
        gram = scale[:, None] * (basis @ np.diag(values) @ basis.T) * scale[None, :]
        score = scale * (basis @ rng.normal(size=q))
        root, resolved = _resolved_root(
            gram, score, scale, count, lambda: (factor, scale, n * q), own=scale
        )

        assert root.shape == (q - 1, q)
        # the computed null vector is the exact one to within Wedin's sin(theta) <=
        # ||dF|| / gap, the QR's gamma_(n q) ||rows|| (Higham 2002, Thm 19.4) over
        # the gap 1e-3 to the next singular value
        angle = _gamma_up(n * q) * np.sqrt(q) * np.linalg.norm(rows, 2) / 1e-3
        direction = null / scale  # the null direction in the Gram's own coordinates
        assert np.linalg.norm(root @ direction) <= angle * np.linalg.norm(root / scale[None, :], 2)
        assert abs(resolved @ direction) <= angle * np.linalg.norm(score / scale)


def _aliased_slope_fit(with_x):
    """A Gaussian fit of a convex spline of x, beside x itself when ``with_x``."""
    from superglm import Numeric, PSpline

    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.uniform(0, 1, n)
    y = 0.2 + 2 * x + 0.6 * x * x + 0.02 * rng.normal(size=n)
    weights = rng.uniform(0.3, 3, n)
    features = {"s": PSpline(n_knots=4, constraint=Constraint.fit.convex)}
    if with_x:
        features = {"x": Numeric(), **features}
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features=features)
    frame = pd.DataFrame({"x": x, "s": x})
    model.fit_reml(frame, y, sample_weight=weights)
    return model, model.predict(frame)


def test_an_ordinary_column_aliased_with_a_scop_slope_changes_nothing():
    """x beside a convex spline of x adds no direction: the Schur block's alias is null, not curvature.

    The alias is an exact zero of ``G~ - C' A^-1 C``, and the square root of
    its rounding sat at the factor's sqrt(eps) cut: read as curvature, it
    sent the inner fits through refused 100-iteration runs and the REML
    search stopped unconverged at lambda 6804 (edf 3.0006) on these rows.
    """
    aliased, aliased_mu = _aliased_slope_fit(with_x=True)
    alone, alone_mu = _aliased_slope_fit(with_x=False)

    assert aliased.reml_diagnostics()["converged"]
    assert aliased.result.converged
    # the outer search's own resolution in log lambda
    assert abs(np.log(aliased._reml_lambdas["s"] / alone._reml_lambdas["s"])) <= 1e-6
    # two certified modes of one model (the certificate's bar, sqrt(r eps), each)
    bar = 2 * np.sqrt((alone.result.beta.size + 2) * 2 * UNIT)
    assert np.max(np.abs(aliased_mu - alone_mu) / np.abs(alone_mu)) <= bar
    assert abs(aliased.result.effective_df - alone.result.effective_df) <= bar * (
        alone.result.beta.size + 2
    )


@pytest.mark.parametrize(
    ("trend", "split"), [(0.0, 1e-7), (0.5, 5e-8)], ids=["independent", "correlated"]
)
def test_nearly_collinear_ordinary_columns_keep_every_scop_direction(trend, split):
    """b = a + split v is full rank, so every SCOP direction is the data's and the fit reaches it.

    The pair's near-collinearity inflated the Gram's formation bound past
    genuine curvature: all but two of the seven Schur directions were
    dropped (the smallest eigenvalue, 0.0287, among them), and the fit
    reported convergence after two iterations at a training RMSE of 0.058.
    With v correlated with s, the rows' rounding bound grows with the pair's
    cancelling coefficients, and a sqrt(eps) cut taken in its units held two
    genuine directions (training RMSE 0.0044); in the columns' own scale they
    are a tenth of their columns or more.
    """
    from superglm import Numeric, PSpline

    rng = np.random.default_rng(2)
    n = 300
    s = rng.uniform(0, 1, n)
    a = rng.normal(size=n)
    v = trend * (s - s.mean()) / s.std() + rng.normal(size=n)
    b = a + split * v
    y = 0.2 + 2 * s + 0.6 * s * s + 0.4 * v
    frame = pd.DataFrame({"a": a, "b": b, "s": s})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.0,
        features={
            "a": Numeric(),
            "b": Numeric(),
            "s": PSpline(n_knots=4, constraint=Constraint.fit.increasing),
        },
    )
    model.fit(frame, y)

    mu = np.asarray(model.predict(frame))
    beta, alpha = np.asarray(model.result.beta), float(model.result.intercept)
    ordinary = beta[0] * a + beta[1] * b
    # y is exact in the model up to b's rounding times beta_b, so the floor is the
    # predictor's evaluation: gamma_(p+2) (|alpha| + sum_j |beta_j x_j|) per row
    # (Higham 2002, sec. 3.1), the spline's terms all of one sign
    size = np.abs(alpha) + np.abs(beta[0] * a) + np.abs(beta[1] * b) + np.abs(mu - alpha - ordinary)
    gamma = (beta.size + 2) * UNIT / (1 - (beta.size + 2) * UNIT)
    assert np.sqrt(np.mean((mu - y) ** 2)) <= gamma * np.sqrt(np.mean(size**2))


@pytest.mark.slow
def test_a_group_beside_a_nearly_collinear_pair_holds_no_other_groups_direction(monkeypatch):
    """Two increasing splines beside ``a, a + 5e-8 v``, v correlated with the first: the rows keep every direction.

    The design is full rank (the ordinary block resolves the pair, and each
    basis is), so every Schur direction is the data's. The first group's
    columns draw on the pair's cancelling coefficients, with rounding bounds
    4e6 to 1.4e7 times their norms; the second group's are at most 1.9e5.
    One bar for every direction, the bounds' 2-norm (0.039 at the first step),
    held a direction made of the second group's columns alone, at 0.038 of
    their scale, whose own rounding is 5e-4 (Golub, Klema & Stewart 1977,
    sec. 3: the bound along a direction is the bounds of the columns it
    draws on).
    """
    from superglm import Numeric, PSpline
    from superglm.solvers import scop_newton

    captured, roots = [], []
    rows_of, resolved_of = SCOPJointNewton._schur_rows, scop_newton._resolved_root

    def capture(self, *args):
        captured.append(rows_of(self, *args))
        return captured[-1]

    def resolve(gram, *args, own):
        calls = len(captured)
        root, score = resolved_of(gram, *args, own=own)
        if len(captured) > calls:  # this decision was the rows'
            roots.append((root.shape[0], gram.shape[0]))
        return root, score

    monkeypatch.setattr(SCOPJointNewton, "_schur_rows", capture)
    monkeypatch.setattr(scop_newton, "_resolved_root", resolve)
    rng = np.random.default_rng(2)
    n = 100_000
    s, t, a = rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.normal(size=n)
    v = 0.5 * (s - s.mean()) / s.std() + rng.normal(size=n)
    y = 0.2 + 2 * s + 0.6 * s * s + 1.5 * np.sqrt(t) + 0.4 * v + 0.1 * rng.normal(size=n)
    increasing = Constraint.fit.increasing
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.0,
        features={
            "a": Numeric(),
            "b": Numeric(),
            "s": PSpline(n_knots=20, constraint=increasing),
            "t": PSpline(n_knots=40, constraint=increasing),
        },
    )
    model.fit(pd.DataFrame({"a": a, "b": a + 5e-8 * v, "s": s, "t": t}), y)

    assert model.result.converged
    assert roots and all(kept == q for kept, q in roots)


def test_an_alias_beside_an_ill_conditioned_ordinary_block_is_null_on_the_rows(monkeypatch):
    """x beside a convex spline of x and a pair ``a, a + 1e-7 v``: the rows certify the alias null.

    The pair makes the ordinary block near-singular, so one projection
    through the formed cross block leaves ``sqrt(dC' A^-1 dC)`` along the
    alias: 1.3e3 times the rows' rounding bound on these rows, read as
    curvature. The second pass leaves it at 4e-4 of the bound on every step.
    """
    from superglm import Numeric, PSpline
    from superglm.solvers import scop_newton
    from superglm.solvers.rank import _gamma_up

    captured, roots = [], []
    rows_of, resolved_of = SCOPJointNewton._schur_rows, scop_newton._resolved_root

    def capture(self, *args):
        captured.append(rows_of(self, *args))
        return captured[-1]

    def resolve(*args, own):
        calls = len(captured)
        root, score = resolved_of(*args, own=own)
        if len(captured) > calls:  # this decision was the rows'
            roots.append((root, own))
        return root, score

    monkeypatch.setattr(SCOPJointNewton, "_schur_rows", capture)
    monkeypatch.setattr(scop_newton, "_resolved_root", resolve)
    rng = np.random.default_rng(3)
    n = 150
    x = rng.uniform(0, 1, n)
    a = np.random.default_rng(11).normal(size=n)
    v = np.random.default_rng(12).normal(size=n)
    y = 0.2 + 2 * x + 0.6 * x * x + 0.02 * rng.normal(size=n) + 0.3 * v
    weights = rng.uniform(0.3, 3, n)
    frame = pd.DataFrame({"x": x, "s": x, "a": a, "b": a + 1e-7 * v})
    convex = PSpline(n_knots=4, constraint=Constraint.fit.convex)

    def fit(features):
        model = SuperGLM(
            family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features=features
        )
        model.fit(frame, y, sample_weight=weights)
        return model

    aliased = fit({"x": Numeric(), "a": Numeric(), "b": Numeric(), "s": convex})
    assert aliased.result.converged
    assert captured and len(roots) == len(captured)
    for (factor, rounding, count), (root, own) in zip(captured, roots, strict=True):
        q = factor.shape[1]
        norms = np.where(own > 0.0, own, 1.0)
        _, singular, right = np.linalg.svd(factor / norms[None, :])
        # the step keeps the q - 1 directions above the rows' cut and holds the alias,
        # which is within the rows' own rounding along it in the columns' scale
        assert root.shape[0] == q - 1
        assert np.sum(singular > scop_newton._row_cut(right.T, rounding / norms, count)) == q - 1
        assert singular[-1] <= _gamma_up(count) * float(np.abs(right[-1]) @ (rounding / norms))

    alone = fit({"a": Numeric(), "b": Numeric(), "s": convex})
    # two certified modes of one model (the certificate's bar, sqrt(r eps), each)
    bar = 2 * np.sqrt((alone.result.beta.size + 2) * 2 * UNIT)
    mu, alone_mu = aliased.predict(frame), alone.predict(frame)
    assert np.max(np.abs(mu - alone_mu) / np.abs(alone_mu)) <= bar


def test_a_well_conditioned_fit_never_reads_the_rows(monkeypatch):
    """Strongly coupled but full rank: every joint step takes the Gram, none the rows (a count)."""
    calls = {"steps": 0, "rows": 0}
    step, rows_of = SCOPJointNewton.step, SCOPJointNewton._schur_rows

    def counted_step(self, *args, **kwargs):
        calls["steps"] += 1
        return step(self, *args, **kwargs)

    def counted_rows(self, *args):
        calls["rows"] += 1
        return rows_of(self, *args)

    monkeypatch.setattr(SCOPJointNewton, "step", counted_step)
    monkeypatch.setattr(SCOPJointNewton, "_schur_rows", counted_rows)
    frame, y, offset = _correlated_monotone_frame(0.9, n=2000)
    _monotone_model().fit(frame, y, offset=offset)

    assert calls["steps"] > 0
    assert calls["rows"] == 0


def test_a_run_finds_each_exact_support_once(monkeypatch):
    """Two row-level SCOP groups' supports are found once per REML run, not once per fit (a count).

    One group's rows repeat (101 distinct values), so it compresses; the
    other's are almost unique, so it does not. Each was found again on every
    inner fit of the run.
    """
    from superglm.reml import scop_efs
    from superglm.solvers import scop_newton

    builds, runs, fits = [], set(), []
    build = scop_newton.build_exact_scop_support
    fit = scop_efs.fit_irls_direct

    def counted_build(rows):
        builds.append(rows.shape)
        return build(rows)

    def counted_fit(*args, **kwargs):
        fits.append(1)
        if kwargs.get("_scop_run_centring") is not None:
            runs.add(id(kwargs["_scop_run_centring"]))
        return fit(*args, **kwargs)

    monkeypatch.setattr(scop_newton, "build_exact_scop_support", counted_build)
    monkeypatch.setattr(scop_efs, "fit_irls_direct", counted_fit)
    rng = np.random.default_rng(8)
    n = 2000
    frame = pd.DataFrame({"a": np.round(rng.uniform(0, 1, n), 2), "b": rng.uniform(0, 1, n)})
    y = 1.0 + np.log1p(3 * frame["a"]) + 2 * frame["b"] ** 2 + 0.3 * rng.normal(size=n)
    increasing = Constraint.fit.increasing
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=False,
        features={
            "a": Spline(kind="ps", k=6, constraint=increasing),
            "b": Spline(kind="ps", k=6, constraint=increasing),
        },
    )
    model.fit_reml(frame, y.to_numpy())

    assert len(fits) > len(runs) > 0
    assert len(builds) == 2 * len(runs)


def test_a_fit_that_drives_coordinates_to_their_boundary_publishes_finite_inference():
    """x beside an increasing spline of x, unpenalized, weights over e^-3..e^3 (Sol's P1 repro).

    The joint step drives several latent coordinates to their log-space
    boundary (one to -1201, past the map's clip), where they are held. The
    post-fit inference squared their vanishing Jacobians, so EDF and
    dispersion came out NaN and ``fit()`` refused the converged fit; the
    line search's trials past the floating-point range also warned.
    """
    from superglm import Numeric, PSpline

    rng = np.random.default_rng(2)
    x = rng.uniform(size=(2, 200))[0]
    y = 0.2 + 2.2 * x + 0.6 * x * x + 0.05 * rng.normal(size=200)
    weights = np.exp(rng.uniform(-3, 3, 200))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.0,
        discrete=False,
        features={"x": Numeric(), "s": PSpline(n_knots=8, constraint=Constraint.fit.increasing)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(pd.DataFrame({"x": x, "s": x}), y, sample_weight=weights)

    result = model.result
    assert result.converged
    # the intercept and every coefficient at most, each held direction at least one fewer
    assert 1.0 <= result.effective_df <= result.beta.size + 1
    assert np.isfinite(result.phi) and result.phi > 0.0


def test_the_schur_rows_are_formed_a_leaf_at_a_time(monkeypatch):
    """The row factor holds a TSQR leaf of rows, never an ``n x q`` copy (a memory envelope, not a time).

    x beside a convex spline of x is an exact alias, so every joint step
    takes the rows. The factor held the projected rows and their weighted
    copy whole, two ``n x q`` arrays, and failed at a million rows under a
    512 MiB address-space limit that master's fit stayed within.
    """
    import tracemalloc

    from superglm import Numeric, PSpline
    from superglm.solvers import scop_newton

    # leaves of a few rows, so one leaf is far below the n x q the envelope bars
    monkeypatch.setattr(scop_newton, "tsqr_leaf_rows", lambda width: 4 * width)
    peaks = []
    rows_of = SCOPJointNewton._schur_rows

    def traced(self, *args):
        tracemalloc.start()
        try:
            return rows_of(self, *args)
        finally:
            peaks.append((tracemalloc.get_traced_memory()[1], self.n * self.width * 8))
            tracemalloc.stop()

    monkeypatch.setattr(SCOPJointNewton, "_schur_rows", traced)
    rng = np.random.default_rng(3)
    x = np.resize(np.linspace(0.0, 1.0, 100), 5_000)
    y = 0.2 + 2 * x + 0.6 * x * x + 0.02 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        discrete=True,
        features={"x": Numeric(), "s": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )
    model.fit(pd.DataFrame({"x": x, "s": x}), y)

    assert model.result.converged
    assert peaks
    assert all(peak < copy for peak, copy in peaks)


def test_a_leaf_reads_a_dense_block_in_place_and_matches_its_row_subset():
    """A leaf's C-ordered dense block is a view of its rows; every block equals ``row_subset``'s."""
    from superglm.group_matrix import CategoricalGroupMatrix
    from superglm.solvers.scop_newton import _leaf_design

    rng = np.random.default_rng(5)
    n = 50
    ordered = rng.normal(size=(n, 3))
    blocks = [
        DenseGroupMatrix(ordered),
        DenseGroupMatrix(np.asfortranarray(rng.normal(size=(n, 2)))),
        CategoricalGroupMatrix(rng.integers(0, 4, n), 3),
    ]
    design = DesignMatrix(blocks, n=n, p=8)
    leaf, copy = _leaf_design(design, 10, 30), design.row_subset(np.arange(10, 30))

    assert np.shares_memory(leaf.group_matrices[0].M, ordered)
    for got, want in zip(leaf.group_matrices, copy.group_matrices, strict=True):
        assert np.array_equal(got.toarray(), want.toarray())
    for got, want in zip(leaf.group_matrices[:2], copy.group_matrices[:2], strict=True):
        assert got.M.flags.c_contiguous and want.M.flags.c_contiguous


def _leaf_block_design(rng, n):
    """A design with every kind of block a leaf multiplies, and its rows' magnitudes as products read them.

    A one-hot block, a support table, a CSR basis (both factored, ``B R^-1``,
    whose products read ``|B| |R^-1|``), a dense block, a subclass, whose
    products are its own, and a plain CSR block as wide as the products'
    six columns, so that a transposed product still fits its slot.
    """
    import scipy.sparse as sp

    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SupportCompressedSSPGroupMatrix,
    )

    class OwnProducts(CategoricalGroupMatrix):
        __slots__ = ()

        def row_subset(self, idx):
            return OwnProducts(self.codes[idx], self.n_levels)

    bins, starts = rng.integers(0, 12, n), rng.integers(0, 5, n)
    support, support_inv = rng.uniform(0, 1, (12, 6)), rng.normal(size=(6, 5))
    entries = (np.repeat(np.arange(n), 4), (starts[:, None] + np.arange(4)).ravel())
    basis = sp.csr_matrix((rng.uniform(0, 1, 4 * n), entries), shape=(n, 8))
    basis_inv = rng.normal(size=(8, 4))
    entries = (np.repeat(np.arange(n), 3), (rng.integers(0, 4, n)[:, None] + np.arange(3)).ravel())
    plain = sp.csr_matrix((rng.uniform(0, 1, 3 * n), entries), shape=(n, 6))
    blocks = [
        CategoricalGroupMatrix(rng.integers(-1, 5, n), 5),
        SupportCompressedSSPGroupMatrix(support, support_inv, bins),
        SparseSSPGroupMatrix(basis, basis_inv),
        DenseGroupMatrix(rng.normal(size=(n, 2)) + 3.0),
        OwnProducts(rng.integers(0, 4, n), 3),
        SparseGroupMatrix(plain),
    ]
    magnitude = [
        blocks[0].toarray(),
        (support @ np.abs(support_inv))[bins],
        np.asarray(basis @ np.abs(basis_inv)),
        np.abs(blocks[3].M),
        blocks[4].toarray(),
        plain.toarray(),
    ]
    dense = np.hstack([block.toarray() for block in blocks])
    return DesignMatrix(blocks, n=n, p=dense.shape[1]), dense, np.hstack(magnitude)


def test_the_leaf_rows_and_their_factor_are_within_their_rounding_on_weights_spread_over_1e15():
    """The Schur rows' leaf products and their TSQR factor, against bounds from the dimensions and u.

    The weights span 1e15, as a cold SCOP start's do beside a rare
    short-exposure claim. Each leaf row ``s - (x - c)' D`` and each entry of
    ``X~' R`` is within ``gamma_k`` of its terms' magnitudes (Higham 2002,
    sec. 3.5), twice that against a dense product that rounds too. The
    factor ``F`` of the weighted rows is a Householder TSQR's, ``A + dA = Q
    F`` with ``||dA||_F <= gamma ||A||_F`` (Thm 19.4, at ``_schur_rows``'s
    count), read at the orthogonal ``Q`` nearest ``A F'`` (the Procrustes
    problem, Golub & Van Loan 2013, sec. 6.4.1).
    """
    from superglm._roundoff import gamma_up
    from superglm.solvers.rank import streamed_weighted_factor
    from superglm.solvers.scop_newton import _leaf_design, _leaf_product, _leaf_transposed

    rng = np.random.default_rng(11)
    n, q, leaf = 4000, 6, 500
    design, dense, magnitude = _leaf_block_design(rng, n)
    p = design.p
    change = rng.normal(size=(p, q)) * 10.0 ** rng.uniform(-3.0, 3.0, (p, 1))
    centre, scop = rng.normal(size=p), rng.normal(size=(n, q))
    weights = 10.0 ** rng.uniform(0.0, 15.0, n)
    weights[:2] = 1.0, 1e15
    # the transposed product centres the dense block's columns, 14 and 15, alone
    centred = dense.copy()
    centred[:, 14:16] -= centre[14:16]
    exact = scop - (dense - centre) @ change
    size = np.abs(scop) + (magnitude + np.abs(centre)) @ np.abs(change)
    gamma = gamma_up(4 * p + leaf)

    leaves = []
    for start in range(0, n, leaf):
        rows = slice(start, start + leaf)
        part = _leaf_design(design, start, rows.stop)
        formed = scop[rows] - _leaf_product(part, change, centre)
        assert np.all(np.abs(formed - exact[rows]) <= 2 * gamma * size[rows])
        weighted = formed * weights[rows, None]
        moment = _leaf_transposed(part, weighted, centre)
        reach = (magnitude[rows] + np.abs(centre)).T @ np.abs(weighted)
        assert np.all(np.abs(moment - centred[rows].T @ weighted) <= 2 * gamma * reach)
        leaves.append((start, rows.stop, formed))

    factor = streamed_weighted_factor(iter(leaves), weights)
    root = np.sqrt(weights)[:, None]
    target = root * exact
    left, _, right = np.linalg.svd(target @ factor.T, full_matrices=False)
    residual = np.linalg.norm(target - (left @ right) @ factor)
    norm = np.linalg.norm(target)
    formation = np.linalg.norm(root * (2 * gamma * size))
    count = n * q + 2 * (p + 3)
    assert residual <= formation + gamma_up(count) * (norm + formation) + gamma_up(4 * q) * norm


@pytest.mark.parametrize("length", [1_000, 40_000], ids=["chunked", "column-at-a-time"])
def test_a_leaf_holds_no_more_of_a_support_than_its_column_products_did(length):
    """A support's products hold no more than one column's products did, however long it is (tracemalloc).

    A column at a time, ``matvec`` held a support-length vector beside two
    leaf-length ones, and ``rmatvec`` its sums beside a leaf-length copy of
    the column. The 1,000-row support's table is formed and summed four
    columns a pass (``_chunks``); the 40,000-row one, longer than the leaf,
    forms no table and takes its own products. Peaks are traced allocations
    above the result; the call's Python objects take under 8 KiB.
    Mutation check: the whole table at once held ``2 q`` support-length
    columns, and a one-column table of the long support held two.
    """
    import tracemalloc

    from superglm._roundoff import gamma_up
    from superglm.group_matrix import SupportCompressedSSPGroupMatrix
    from superglm.solvers.scop_newton import _leaf_product, _leaf_transposed

    rng = np.random.default_rng(5)
    n, q, k = 4_000, 16, 6
    support, support_inv = rng.uniform(0, 1, (length, k)), rng.normal(size=(k, k - 1))
    bins = rng.integers(0, length, n)
    part = DesignMatrix([SupportCompressedSSPGroupMatrix(support, support_inv, bins)], n=n, p=k - 1)
    change, centre = rng.normal(size=(k - 1, q)), rng.normal(size=k - 1)
    rows = rng.normal(size=(n, q))

    _leaf_product(part, change, centre), _leaf_transposed(part, rows, centre)  # compiled
    tracemalloc.start()
    try:
        formed = _leaf_product(part, change, centre)
        product_peak = tracemalloc.get_traced_memory()[1] - formed.nbytes
        tracemalloc.reset_peak()
        moment = _leaf_transposed(part, rows, centre)
        moment_peak = tracemalloc.get_traced_memory()[1] - formed.nbytes - moment.nbytes
    finally:
        tracemalloc.stop()

    assert product_peak <= 8 * max(length + n, 2 * n) + 8 * 1024
    assert moment_peak <= 8 * (length + n) + moment.nbytes + 8 * 1024
    # each entry within gamma_k of its terms' magnitudes (Higham 2002, sec. 3.5),
    # twice that against a dense product that rounds too
    basis = support @ support_inv
    reach = (support @ np.abs(support_inv) + np.abs(centre)) @ np.abs(change)
    error = np.abs(formed - (basis[bins] - centre) @ change)
    assert np.all(error <= 2 * gamma_up(2 * k) * reach[bins])
    sums = np.zeros((length, q))
    np.add.at(sums, bins, np.abs(rows))
    reach = np.abs(support_inv).T @ (support.T @ sums)
    error = np.abs(moment - basis[bins].T @ rows)
    assert np.all(error <= 2 * gamma_up(n + length + k) * reach)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_a_rare_short_exposure_claim_takes_the_rows_and_each_block_multiplies_all_columns(
    monkeypatch, discrete
):
    """Every joint step takes the rows, and a leaf multiplies each block for all its columns at once (counts).

    Five top levels of an increasing ``x`` hold one claim each at exposure
    0.002, where the intercept-only fit expects 2e-4, so the cold start's
    weights reach ``e^35`` there. The Gram's decision is forced to the rows
    on every joint step (``_GREY_FLOOR``), so the counts do not depend on
    the trajectory reaching a grey step. A leaf multiplies each ordinary
    block's table once for all its columns, in one pass over the rows
    (``_add_gathered``, ``_bin_sums``), and a CSR basis's transposed product
    in one multivector product: never a table's ``matvec`` or ``rmatvec``,
    or a CSR ``rmatvec``, a column at a time. Mutation check: master's leaf
    products (7dcc85c1) made such calls on every row-route call.
    """
    from superglm import Categorical, PSpline
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseSSPGroupMatrix,
    )
    from superglm.solvers import scop_newton

    rng = np.random.default_rng(1)
    n = 3000
    x = np.minimum(rng.geometric(0.15, n) - 1, 60).astype(float)
    z, exposure = rng.uniform(0.0, 1.0, n), rng.uniform(0.05, 1.0, n)
    y = rng.poisson(exposure * 0.1 * np.exp(0.02 * x + 0.3 * np.sin(6.0 * z))).astype(float)
    x[:5], exposure[:5], y[:5] = 60.0 + np.arange(5.0), 0.002, 1.0
    frame = pd.DataFrame({"x": x, "z": z, "g": rng.integers(0, 4, n).astype(str)})

    calls = {"rows": 0, "resolved": 0, "columnwise": 0, "gathered": 0, "summed": 0}
    inside = [False]

    def counted(name, function):
        def wrapper(*args, **kwargs):
            calls[name] += inside[0]
            return function(*args, **kwargs)

        return wrapper

    rows_of = SCOPJointNewton._schur_rows

    def rows(self, *args):
        calls["rows"] += 1
        inside[0] = True
        try:
            return rows_of(self, *args)
        finally:
            inside[0] = False

    monkeypatch.setattr(SCOPJointNewton, "_schur_rows", rows)
    monkeypatch.setattr(scop_newton, "_GREY_FLOOR", np.inf)
    resolved_of = scop_newton._resolved_root

    def resolved(*args, **kwargs):
        calls["resolved"] += 1
        return resolved_of(*args, **kwargs)

    monkeypatch.setattr(scop_newton, "_resolved_root", resolved)
    for kind, method in (
        (CategoricalGroupMatrix, "matvec"),
        (CategoricalGroupMatrix, "rmatvec"),
        (DiscretizedSSPGroupMatrix, "matvec"),
        (DiscretizedSSPGroupMatrix, "rmatvec"),
        (SparseSSPGroupMatrix, "rmatvec"),
    ):
        monkeypatch.setattr(kind, method, counted("columnwise", getattr(kind, method)))
    monkeypatch.setattr(
        scop_newton, "_add_gathered", counted("gathered", scop_newton._add_gathered)
    )
    monkeypatch.setattr(scop_newton, "_bin_sums", counted("summed", scop_newton._bin_sums))
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=discrete,
        features={
            "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
            "z": PSpline(n_knots=6),
            "g": Categorical(),
        },
    )
    model.fit_reml(frame, y, offset=np.log(exposure))

    assert model.result.converged
    assert calls["rows"] == calls["resolved"] > 0
    assert calls["columnwise"] == 0
    assert calls["gathered"] >= 2 * calls["rows"] and calls["summed"] >= calls["rows"]


def test_a_run_finds_a_lone_exact_support_once(monkeypatch):
    """A fit's only row-level SCOP group is compressed once per REML run, not once per fit (a count).

    The lone group takes its own route, compressing whenever a row repeats;
    it read the group's rows and sorted them on every inner fit of the run.
    """
    from superglm import Numeric
    from superglm.reml import scop_efs
    from superglm.solvers import scop_exact_support

    builds, runs, fits = [], set(), []
    build = scop_exact_support.build_exact_scop_support
    fit = scop_efs.fit_irls_direct

    def counted_build(rows):
        builds.append(rows.shape)
        return build(rows)

    def counted_fit(*args, **kwargs):
        fits.append(1)
        if kwargs.get("_scop_run_centring") is not None:
            runs.add(id(kwargs["_scop_run_centring"]))
        return fit(*args, **kwargs)

    monkeypatch.setattr(scop_exact_support, "build_exact_scop_support", counted_build)
    monkeypatch.setattr(scop_efs, "fit_irls_direct", counted_fit)
    rng = np.random.default_rng(8)
    n = 2000
    frame = pd.DataFrame({"a": np.round(rng.uniform(0, 1, n), 2), "z": rng.normal(size=n)})
    y = 1.0 + np.log1p(3 * frame["a"]) + 0.5 * frame["z"] + 0.3 * rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=False,
        features={
            "a": Spline(kind="ps", k=6, constraint=Constraint.fit.increasing),
            "z": Numeric(),
        },
    )
    model.fit_reml(frame, y.to_numpy())

    assert len(fits) > len(runs) > 0
    assert len(builds) == len(runs)


def test_an_ordinary_column_within_sqrt_eps_of_a_scop_slope_is_held_not_chased():
    """x + 1e-9 v beside a convex spline of x: their split is below the Pya-Wood cut, so it is held.

    The rows resolve the pair above their rounding (3.5e-9 of its column
    scale), so the step chased the split between x and the spline's free
    slope: 3.5e6 along it, cancelling against the ordinary correction, until
    the line search refused the fit after four iterations, its deviance 1.8%
    above the block-coordinate fit's. Below sqrt(eps) of its scale the
    split is the ordinary design's (``_resolved_root``), and the fit is the
    spline's alone.
    """
    from superglm import Numeric, PSpline

    rng = np.random.default_rng(31)
    n = 2000
    x = rng.uniform(0, 1, n)
    y = 0.2 + 2.2 * x + 0.6 * x * x + 0.1 * rng.normal(size=n)
    frame = pd.DataFrame({"x": x + 1e-9 * rng.normal(size=n), "s": x})

    def fit(features):
        model = SuperGLM(family="gaussian", selection_penalty=0.0, tol=1e-12, features=features)
        model.fit(frame, y)
        return model

    convex = PSpline(n_knots=8, constraint=Constraint.fit.convex)
    near, alone = fit({"x": Numeric(), "s": convex}), fit({"s": convex})

    assert near.result.converged
    # two modes of one model, each stopped within sqrt(r eps) (the certificate's bar)
    bar = 2 * np.sqrt((alone.result.beta.size + 2) * 2 * UNIT)
    mu, alone_mu = near.predict(frame), alone.predict(frame)
    assert np.max(np.abs(mu - alone_mu) / np.abs(alone_mu)) <= bar


def _ordinary_inside_the_spline_fit():
    """x beside an increasing spline of the same x, weights exp(U(-3, 3)), no penalty: Gaussian."""
    from superglm import Numeric, PSpline

    rng = np.random.default_rng(39)
    x = rng.uniform(size=(2, 200))[0]
    y = 0.2 + 2.2 * x + 0.6 * x * x + 0.05 * rng.normal(size=200)
    weights = np.exp(rng.uniform(-3, 3, 200))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.0,
        features={"x": Numeric(), "s": PSpline(n_knots=8, constraint=Constraint.fit.increasing)},
    )
    frame = pd.DataFrame({"x": x, "s": x})
    model.fit(frame, y, sample_weight=weights)
    return model, frame, x, y, weights


def test_a_damped_joint_step_never_stops_the_fit():
    """x inside an increasing spline's span: the fit stops on a full step, at the ordinary block's mode.

    A latent coordinate at the QP start's floor wants to grow, so its
    Gauss-Newton step is millions long in log space and the line search
    halves the joint step to 2^-20. The deviance changed by that fraction,
    and the fit stopped there with intercept and slope scores -1.7 and -5.2.
    A full step's decrease is, to second order, its Newton decrement ``s'
    H^-1 s``, which bounds each score by ``sqrt(H_jj s' H^-1 s)``; the stop
    holds that decrease within ``tol`` of the deviance.
    """
    model, frame, x, y, weights = _ordinary_inside_the_spline_fit()
    residual = y - np.asarray(model.predict(frame))
    deviance = float(np.sum(weights * residual**2))

    assert model.result.converged
    for column in (np.ones_like(x), x):
        score = float(np.sum(weights * residual * column))
        assert abs(score) <= np.sqrt(np.sum(weights * column**2) * model._tol * deviance)


def _latent_step_depth_of(*changes, free_dim=2):
    """``_latent_step_depth`` of a trial moving group ``i`` by ``changes[i]``, its first ``free_dim`` coordinates free."""

    def trial(betas):
        groups = tuple(
            irls_direct._SCOPGroupState(i, beta, np.zeros_like(beta), None, 0.0, False)
            for i, beta in enumerate(betas)
        )
        return irls_direct._SCOPTrialState(irls=None, groups=groups)

    moved = [np.asarray(change, dtype=float) for change in changes]
    specs = {
        i: SimpleNamespace(reparam=SimpleNamespace(free_dim=free_dim)) for i in range(len(moved))
    }
    return irls_direct._latent_step_depth(
        trial([np.zeros_like(m) for m in moved]), trial(moved), specs
    )


@pytest.mark.parametrize(
    ("changes", "depth"),
    [
        pytest.param([[1e12, -1e12, 0, 0, 0]], 0, id="free-coordinates-are-not-bounded"),
        pytest.param([[0, 1e300, 0, 0, 0]], 0, id="last-free-coordinate-is-not-bounded"),
        pytest.param([[0, 0, 1000.0, 0, 0]], 0, id="first-exponentiated-at-the-maximum"),
        pytest.param([[0, 0, 0, 0, -1000.0]], 0, id="a-decrease-counts-by-its-length"),
        pytest.param([[0, 0, 0, 1000.4, 0]], 1, id="just-past-the-maximum"),
        pytest.param([[0, 0, 2000.0, 0, 0]], 1, id="halved-onto-the-maximum"),
        pytest.param([[0, 0, 2000.5, 0, 0]], 2, id="halved-just-past-the-maximum"),
        pytest.param([[0, 0, 0, 4000.0, 0]], 2, id="quartered-onto-the-maximum"),
        pytest.param([[0, 0, 0, 0, 6.6e7]], 17, id="the-repro-step"),
        pytest.param([[0, 0, 0, np.inf, 0]], 0, id="an-infinite-step-is-left-to-the-search"),
        pytest.param([[0, 0, 0, 0, -np.inf]], 0, id="a-negative-infinite-step"),
        pytest.param([[0, 0, 10.0, 0, 0], [0, 0, 0, 4000.5, 0]], 3, id="the-longest-group-last"),
        pytest.param([[0, 0, 4000.5, 0, 0], [0, 0, 0, 10.0, 0]], 3, id="the-longest-group-first"),
        pytest.param([[0, 0, 0, np.inf, 0], [0, 0, 0, 4000.5, 0]], 0, id="infinite-in-any-group"),
    ],
)
def test_the_latent_step_depth_is_the_first_halving_within_the_maximum_step(changes, depth):
    """The smallest ``d`` with ``max |step| / 2^d <= _LATENT_MAX_STEP`` over the exponentiated coordinates.

    A step of exactly the maximum is within it (depth 0), a free coordinate's
    step is not counted whatever its length, and a step that is not finite is
    depth 0: no halving brings it back, and the line search refuses it as it is.
    """
    assert irls_direct._LATENT_MAX_STEP == 1000.0
    assert _latent_step_depth_of(*changes) == depth


def test_a_latent_step_past_the_maximum_step_is_searched_from_within_it(monkeypatch):
    """x inside a convex spline's span, weights exp(U(-3, 3)), no penalty (Sol's round-5 repro).

    A latent coordinate at the QP start's floor wants to grow, and its
    Newton step is 6.6e7 long in log space. Every trial within the line
    search's 20 halvings left it past the map's clip, so the step was
    rejected on the first iteration (weighted RSS 29.98, scores 55 and 60).
    The search now starts from the step's first halving within
    ``_LATENT_MAX_STEP`` and tries nothing longer; the fit stops on a full
    step, each ordinary score within its Newton-decrement bound (as in
    ``test_a_damped_joint_step_never_stops_the_fit``).
    """
    from superglm import Numeric, PSpline

    lengths, depths = [], []
    evaluate = irls_direct._evaluate_scop_trial
    depth_of = irls_direct._latent_step_depth

    def recorded(**kwargs):
        trial = evaluate(**kwargs)
        for before, after in zip(kwargs["committed"].groups, trial.groups, strict=True):
            free = kwargs["specs"][before.group_index].reparam.free_dim
            lengths.append(np.max(np.abs(after.beta_eff - before.beta_eff)[free:], initial=0.0))
        return trial

    def recorded_depth(*args):
        depths.append(depth_of(*args))
        return depths[-1]

    monkeypatch.setattr(irls_direct, "_evaluate_scop_trial", recorded)
    monkeypatch.setattr(irls_direct, "_latent_step_depth", recorded_depth)
    rng = np.random.default_rng(11)
    x = rng.uniform(size=200)
    y = 0.2 + 2.2 * x + 0.6 * x * x + 0.05 * rng.normal(size=200)
    weights = np.exp(rng.uniform(-3, 3, 200))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.0,
        features={"x": Numeric(), "s": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )
    frame = pd.DataFrame({"x": x, "s": x})
    model.fit(frame, y, sample_weight=weights)
    residual = y - np.asarray(model.predict(frame))
    deviance = float(np.sum(weights * residual**2))

    # the bound below is the shortening's, not a trajectory that never needed it
    assert any(depths)
    assert lengths and max(lengths) <= irls_direct._LATENT_MAX_STEP
    assert model.result.converged
    for column in (np.ones_like(x), x):
        score = float(np.sum(weights * residual * column))
        assert abs(score) <= np.sqrt(np.sum(weights * column**2) * model._tol * deviance)


def test_a_damped_trial_takes_the_ordinary_solve_at_its_latent_coordinates(monkeypatch):
    """Each trial's ordinary scores are ``1 - alpha`` times the committed iterate's (Gaussian, identity).

    Given the latent coordinates the ordinary block is linear. A trial takes
    its solve at the trial's own mapped coefficients, the part not owed to
    the latent change damped along ``alpha``, and the scores are linear in
    the distance to that solve: they are ``1 - alpha`` times the committed
    ones, however far the latent step goes. Interpolating the ordinary block
    along the joint step's linearisation of ``exp`` leaves the latent map's
    curvature in them, which a step millions long in log space makes large.
    The bound is each predictor's evaluation, ``gamma_(n)`` of its terms,
    carried through the scores' sums.
    """
    trials = []
    evaluate = irls_direct._evaluate_scop_trial

    def recorded(**kwargs):
        trial = evaluate(**kwargs)
        trials.append((kwargs["alpha"], kwargs["committed"].irls, trial.irls))
        return trial

    monkeypatch.setattr(irls_direct, "_evaluate_scop_trial", recorded)
    model, frame, x, y, weights = _ordinary_inside_the_spline_fit()
    design = model._dm.toarray()
    columns = np.column_stack([np.ones_like(x), x])
    gamma = (y.size + design.shape[1] + 2) * UNIT / (1 - (y.size + design.shape[1] + 2) * UNIT)

    checked = 0
    for alpha, before, after in trials:
        if not np.all(np.isfinite(after.eta_unclipped)):
            continue
        terms = sum(
            abs(state.intercept) + np.abs(design) @ np.abs(state.beta) for state in (before, after)
        )
        bound = gamma * (np.abs(columns).T @ (weights * (np.abs(y) + terms)))
        scores = columns.T @ (weights * (y - after.eta_unclipped))
        committed = columns.T @ (weights * (y - before.eta_unclipped))
        assert np.all(np.abs(scores - (1 - alpha) * committed) <= bound)
        checked += alpha <= 2.0**-10
    assert checked


_COUPLED_MONOTONE_TOL = 1e-6


def _fit_coupled_monotone(convergence):
    """The corr(x1, x2) = 0.7 monotone fixture's inner fit at fixed lambda, stopping on ``convergence``."""
    frame, y, offset = _correlated_monotone_frame(0.7)
    model = _monotone_model()
    y_fit, weights, offset_fit = model_build_design_matrix(model, frame, y, np.ones_like(y), offset)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y_fit,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x1": 8.94, "x2": 0.770},
        offset=offset_fit,
        max_iter=100,
        tol=_COUPLED_MONOTONE_TOL,
        weight_semantics="frequency",
        convergence=convergence,
    )
    return result


@pytest.mark.parametrize("convergence", ["deviance", "coefficients"])
def test_a_damped_step_is_never_where_the_fit_stops(monkeypatch, convergence):
    """A step the line search cut to 2^-20 changes the deviance, and the coefficients, by about that fraction: the fit goes on.

    SCOP REML inner fits stop on the coefficients (``scop_efs``).
    """
    from superglm.solvers.irls_state import _IRLSStepDecision

    decisions = []
    select = irls_direct._select_irls_trial

    def damp_the_second(**kwargs):
        decisions.append(kwargs)
        if len(decisions) == 2:
            kwargs["evaluate_state"](2.0**-20)
            return _IRLSStepDecision(2.0**-20, 20, False, trials_attempted=2)
        return select(**kwargs)

    monkeypatch.setattr(irls_direct, "_select_irls_trial", damp_the_second)
    result = _fit_coupled_monotone(convergence)

    assert result.converged
    assert result.n_iter > 2


@pytest.mark.parametrize("convergence", ["deviance", "coefficients"])
def test_a_step_shortened_into_the_latent_range_is_never_where_the_fit_stops(
    monkeypatch, convergence
):
    """A proposal cut to 2^-30 of the Newton step by the latent range, then taken whole by the search, is not a stop.

    The second iteration's ``_latent_step_depth`` is forced to 30 and
    ``_select_irls_trial`` is left alone, so the search accepts the shortened
    proposal at its own alpha = 1: its decision reports no halving, and only
    the depth added to ``step_halvings`` tells the stop rules the step is
    damped. Without it the change in deviance, and in the coefficients, is
    that same 2^-30 of a full step's and the fit stops on it at iteration 2.
    The depth is deep enough that this change is below ``tol``; a depth of
    3 is not, and the fit would go on whether or not the depth is counted.
    The retained iterate is the trial evaluated at 2^-30, not the full
    proposal the search never tried.
    """
    depth = 30
    states, trials, decisions = [], [], []
    depth_of = irls_direct._latent_step_depth
    evaluate = irls_direct._evaluate_scop_trial
    select = irls_direct._select_irls_trial

    def shortened_second(committed, proposed, specs):
        states.append((committed, proposed))
        return depth if len(states) == 2 else depth_of(committed, proposed, specs)

    def recorded_trial(**kwargs):
        trial = evaluate(**kwargs)
        trials.append((len(states), kwargs["alpha"], trial))
        return trial

    def recorded_decision(**kwargs):
        decision = select(**kwargs)
        decisions.append((len(states), decision))
        return decision

    monkeypatch.setattr(irls_direct, "_latent_step_depth", shortened_second)
    monkeypatch.setattr(irls_direct, "_evaluate_scop_trial", recorded_trial)
    monkeypatch.setattr(irls_direct, "_select_irls_trial", recorded_decision)
    result = _fit_coupled_monotone(convergence)

    assert result.converged
    assert result.n_iter > 2

    # the event: the search took the shortened proposal at its own alpha = 1
    (forced,) = [decision for iteration, decision in decisions if iteration == 2]
    assert (forced.alpha, forced.step_halvings, forced.step_rejected) == (1.0, 0, False)
    (shortened,) = [
        trial for iteration, alpha, trial in trials if iteration == 2 and alpha == 2.0**-depth
    ]
    committed, proposed = states[1]
    retained = states[2][0]
    latent_change = 0.0
    for before, full, trial, after in zip(
        committed.groups, proposed.groups, shortened.groups, retained.groups, strict=True
    ):
        assert np.array_equal(after.beta_eff, trial.beta_eff)
        assert not np.array_equal(after.beta_eff, full.beta_eff)
        move = np.abs(after.beta_eff - before.beta_eff) / np.maximum(1.0, np.abs(after.beta_eff))
        latent_change = max(latent_change, float(np.max(move)))
    # the premise: a stop read off this step's change would have fired
    assert latent_change < _COUPLED_MONOTONE_TOL


def _sparse_counts_model(n, seed, theta):
    """Counts of mean about 0.05 beside an increasing spline: NB2 data at the Poisson limit."""
    from superglm import PSpline
    from superglm.distributions import NegativeBinomial

    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(0, 10, n), rng.uniform(-2, 2, n)
    exposure = rng.uniform(0.2, 1.0, n)
    y = rng.poisson(np.exp(0.06 * x1 + 0.3 * np.sin(1.3 * x2) - 3.0) * exposure)
    model = SuperGLM(
        family=NegativeBinomial(theta=theta),
        features={
            "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
            "x2": Spline(kind="cr", k=8),
        },
    )
    return model, pd.DataFrame({"x1": x1, "x2": x2}), y.astype(float), np.log(exposure)


def test_a_trial_past_binary64s_range_is_refused_without_a_warning(monkeypatch):
    """NB2 with theta estimated on sparse counts (the release differential's NB2 SCOP case, reduced).

    A latent coordinate at the QP start's floor takes a step past
    ``_LATENT_MAX_STEP``, whose first halving within it maps to about
    exp(500); the trial's ordinary block, solved at those mapped
    coefficients, is as large, and its penalty's quadratic overflowed in
    the merit, warning on a trial the line search refuses anyway.
    """
    largest = []
    evaluate = irls_direct._evaluate_scop_trial

    def recorded(**kwargs):
        trial = evaluate(**kwargs)
        largest.append(float(np.max(np.abs(trial.irls.beta))))
        return trial

    monkeypatch.setattr(irls_direct, "_evaluate_scop_trial", recorded)
    model, frame, y, offset = _sparse_counts_model(400, 2, "auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(frame, y, offset=offset)

    assert max(largest) > np.sqrt(np.finfo(float).max)
    assert model.result.converged


def test_an_nb2_fit_at_the_poisson_limit_stops_on_a_full_step():
    """NB2 at theta = 1e8 on sparse counts, under a REML inner fit's criterion (coefficients, 1e-10).

    NB2's unit deviance rounds at about u theta a row whatever its size, so
    near the mode one row's rounding moving by an ulp moved the merit by
    4.4e-8, a hundred times its allowance: every full step was refused and,
    a damped step never stopping the fit, it ran out its 100 iterations (a
    REML fit refuses such a fit, as the release differential's NB2 SCOP case
    did). The allowance now carries each row's 2 gamma_4 (2y + theta), which
    bounds the unit deviance's error against its log1p form.
    """
    from scipy.special import xlogy

    model, frame, y, offset = _sparse_counts_model(400, 3, 1e8)
    y_fit, weights, offset_fit = model_build_design_matrix(model, frame, y, np.ones_like(y), offset)
    result, _ = irls_direct.fit_irls_direct(
        model._dm,
        y_fit,
        weights,
        model._distribution,
        model._link,
        model._groups,
        lambda2={"x1": 10.0, "x2": 4.4},
        offset=offset_fit,
        max_iter=100,
        tol=1e-10,
        weight_semantics="frequency",
        convergence="coefficients",
    )

    assert result.converged
    theta, mu = 1e8, np.geomspace(1e-3, 3.0, 999)
    gamma4 = 4 * UNIT / (1 - 4 * UNIT)
    for count in (0.0, 1.0, 3.0):
        first = xlogy(count, count / mu)
        second = (count + theta) * np.log1p((count - mu) / (mu + theta))
        error = model._distribution.deviance_unit(np.full_like(mu, count), mu) - 2 * (
            first - second
        )
        terms = np.abs(first) + np.abs(second)
        assert np.all(np.abs(error) <= 2 * gamma4 * (2 * count + theta + terms))
