"""SCOP admission uses represented quadratic actions, not absolute units."""

import numpy as np
import pytest

from superglm.distributions import Gaussian
from superglm.group_matrix import DesignMatrix
from superglm.links import IdentityLink
from superglm.solvers.centered_system import build_centered_system
from superglm.solvers.irls_state import _irls_objective_relative_change
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import SCOPGroupSpec, SCOPJointNewton
from superglm.types import GroupSlice


@pytest.mark.parametrize("scale", [1e-24, 1e-20, 1.0, 1e20, 1e24])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("penalty", [0.0, 0.5])
def test_scop_step_is_invariant_to_the_objective_scale(scale, groups, discrete, penalty):
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    design = np.kron(np.eye(groups), np.array([[-1.0], [1.0]]))
    response = np.tile([-10.0, 10.0], groups)
    weights = np.full(len(response), scale)
    n = len(response)
    bin_idx = np.arange(n) if discrete else None
    specs = [
        SCOPGroupSpec(
            group_index=i,
            group=GroupSlice(str(i), i, i + 1),
            reparam=reparam,
            B_scop=design[:, i : i + 1],
            S_scop=np.eye(1),
            bin_idx=bin_idx,
        )
        for i in range(groups)
    ]
    empty = DesignMatrix([], n=n, p=0)
    engine = SCOPJointNewton(specs, scale * penalty, empty, n)
    thetas = [np.zeros(1)] * groups
    reduced = build_centered_system(
        dm=empty, W=weights, z_off=response - engine.predictor(thetas), penalty=np.zeros((0, 0))
    )
    step = engine.step(
        thetas,
        W=weights,
        z=response,
        eta=engine.predictor(thetas),
        reduced=reduced,
        solve=lambda column: column,
        ordinary_change=np.zeros(0),
    )

    # The map curvature makes the block indefinite, and the Gauss-Newton
    # direction is 18 / (2 + penalty) in every coordinate at every scale.
    beta = np.concatenate([step.results[i].beta_new for i in range(groups)])
    assert all(result.used_fisher_fallback for result in step.results.values())
    allowance = 128 * np.finfo(float).eps
    np.testing.assert_allclose(beta, np.full(groups, 18.0 / (2.0 + penalty)), rtol=allowance)


@pytest.mark.parametrize("scale", [1e-24, 1.0, 1e24])
def test_actual_scop_outer_boundary_uses_weighted_objective_units(scale):
    """The Newton proposal (beta_eff 9) overshoots, and the outer search halves it twice."""
    import superglm.solvers.irls_direct as irls_direct

    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    result, _, states = irls_direct.fit_irls_direct(
        np.array([[-1.0], [1.0]]),
        np.array([-10.0, 10.0]),
        np.full(2, scale),
        Gaussian(),
        IdentityLink(),
        [GroupSlice("shape", 0, 1, monotone_engine="scop", scop_reparameterization=reparam)],
        lambda2=0.0,
        beta_init=np.ones(1),
        intercept_init=0.0,
        max_iter=1,
        S_override=np.zeros((1, 1)),
        return_scop_state=True,
        scop_state_init={0: {"beta_eff": np.zeros(1), "S_scop": np.zeros((1, 1))}},
        record_diagnostics=True,
        _compute_scop_postfit_inference=False,
        weight_semantics="frequency",
    )
    assert not result.converged
    assert result.n_iter == 1
    assert result.deviance / scale < 162.0
    expected_change = abs(result.deviance - 162.0 * scale) / (162.0 * scale)
    assert result.iteration_log[0].convergence_value == pytest.approx(
        expected_change, rel=16 * np.finfo(float).eps, abs=0.0
    )
    np.testing.assert_allclose(states[0]["beta_eff"], [2.25], rtol=128 * np.finfo(float).eps)
    assert result.iteration_log[0].step_halvings == 2


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e308])
def test_objective_change_ratio_preserves_finite_extreme_units(scale):
    ratio = _irls_objective_relative_change(
        objective=0.9 * scale, previous=scale, objective_scale=scale
    )
    assert ratio == pytest.approx(0.05, rel=16 * np.finfo(float).eps, abs=0.0)


def test_objective_change_ratio_has_explicit_zero_action_semantics():
    assert _irls_objective_relative_change(objective=0.0, previous=0.0, objective_scale=0.0) == 0.0
    assert np.isinf(
        _irls_objective_relative_change(objective=1.0, previous=0.0, objective_scale=0.0)
    )
