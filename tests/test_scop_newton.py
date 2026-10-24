"""The joint SCOP Newton step (Pya & Wood 2015, Appendix 1)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from superglm import Constraint, Spline, SuperGLM
from superglm.distributions import Tweedie
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.model.base import model_build_design_matrix
from superglm.solvers import irls_direct
from superglm.solvers.centered_system import build_centered_system
from superglm.solvers.rank import decompose_gram_if_authoritative
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import SCOPGroupSpec, SCOPJointNewton
from superglm.types import GroupSlice

UNIT = np.finfo(np.float64).eps / 2.0


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
    for result in step.results.values():
        # the curvature the REML geometry reads is the one the step solved with
        assert np.linalg.eigvalsh(result.H_penalized).min() > 0.0


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


def test_tweedie_log_scop_fit_takes_observed_newton_rows():
    """The joint step is full Newton: Tweedie/log rows carry the observed information.

    On Fisher rows a Tweedie/log SCOP inner fit contracts only linearly where
    the observed information exceeds Fisher's, as heavy-tailed pure premiums
    give: 0.8 to 0.94 per iteration on 50,000 freMTPL2 rows, where most inner
    fits exhausted their 100 iterations.
    """
    rng = np.random.default_rng(5)
    n = 4000
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
    profile: dict = {}
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

    assert result.converged
    assert profile.get("irls_observed_newton_iters", 0) == result.n_iter


def test_one_pass_cross_block_equals_the_per_pair_plan():
    """The joint-weight route forms X_r' W X_s as the group-matrix kernels do, to rounding."""
    from superglm import Categorical
    from superglm.group_matrix import (
        CrossMatrixExecutionPlan,
        DiscretizedSCOPGroupMatrix,
        MatrixExecutionPlan,
    )
    from superglm.solvers.scop_newton import _JointWeightCross

    rng = np.random.default_rng(11)
    n = 5000
    frame = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, n),
            "b": rng.uniform(0, 1, n),
            "m": rng.uniform(0, 1, n),
            "c": rng.choice(list("pqrst"), n),
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
    scop = next(gm for g, gm in pairs if g.monotone_engine == "scop")
    ordinary = [gm for g, gm in pairs if g.monotone_engine != "scop"]
    reduced = DesignMatrix(ordinary, n=n, p=sum(gm.shape[1] for gm in ordinary))
    weights = rng.uniform(0.1, 3.0, n)
    plan = CrossMatrixExecutionPlan(
        reduced.execution_plan,
        MatrixExecutionPlan([DiscretizedSCOPGroupMatrix(scop.B_scop_unique, scop.bin_idx)], n=n),
    )

    one_pass = _JointWeightCross.build(reduced, [(scop.B_scop_unique, scop.bin_idx)])
    expected = plan.cross_moment(weights)
    # sums of the same n products in another order: within n u of the largest entry
    np.testing.assert_allclose(
        one_pass.cross_moment(weights), expected, rtol=0, atol=n * UNIT * np.abs(expected).max()
    )
    # a group with no compact support (a dense block) keeps the per-pair plan
    dense = DesignMatrix(
        [*ordinary, DenseGroupMatrix(frame[["a"]].to_numpy())], n=n, p=reduced.p + 1
    )
    assert _JointWeightCross.build(dense, [(scop.B_scop_unique, scop.bin_idx)]) is None


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
    plan = CrossMatrixExecutionPlan(
        reduced.execution_plan,
        MatrixExecutionPlan([DiscretizedSCOPGroupMatrix(scop.B_scop_unique, scop.bin_idx)], n=n),
    )
    expected = plan.cross_moment(weights)
    # sums of the same n products in another order: within n u of the largest entry
    np.testing.assert_allclose(
        one_pass.cross_moment(weights), expected, rtol=0, atol=n * UNIT * np.abs(expected).max()
    )

    # Tables past the cap are refused from the groups' metadata, before any support is formed.
    formed.clear()
    monkeypatch.setattr(scop_newton, "_MAX_PACKED_HIST_CELLS", levels * scop.B_scop_unique.shape[0])
    fresh = DesignMatrix(ordinary, n=n, p=reduced.p)
    assert scop_newton._JointWeightCross.build(fresh, columns) is None
    assert formed == []
