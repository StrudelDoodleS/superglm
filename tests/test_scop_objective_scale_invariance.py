"""SCOP admission uses represented quadratic actions, not absolute units."""

import numpy as np
import pytest

from superglm.distributions import Gaussian
from superglm.links import IdentityLink
from superglm.solvers.irls_state import _irls_objective_relative_change
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import (
    SCOPNewtonResult,
    scop_joint_newton_step,
    scop_newton_step,
)
from superglm.types import GroupSlice


@pytest.mark.parametrize("scale", [1e-24, 1e-20, 1.0, 1e20, 1e24])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("penalty", [0.0, 0.5])
def test_scop_step_decreases_the_scaled_quadratic(scale, joint, discrete, penalty):
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    group_count = 2 if joint else 1
    design = np.kron(np.eye(group_count), np.array([[-1.0], [1.0]]))
    response = np.tile([-10.0, 10.0], group_count)
    weights = np.full(len(response), scale)
    bin_idx = np.arange(len(response)) if discrete else None
    if joint:
        groups = [GroupSlice(str(i), i, i + 1) for i in range(group_count)]
        states = {
            i: {
                "B_scop": design[:, i : i + 1],
                "S_scop": np.eye(1),
                "beta_scop": np.zeros(1),
                "reparam": reparam,
                "bin_idx": bin_idx,
            }
            for i in range(group_count)
        }
        results = scop_joint_newton_step(states, weights, response, scale * penalty, groups)
        beta = np.concatenate([results[i].beta_new for i in range(group_count)])
        reported = results[0]
    else:
        reported = scop_newton_step(
            design,
            weights,
            response,
            np.zeros(1),
            reparam,
            np.eye(1),
            scale * penalty,
            bin_idx=bin_idx,
        )
        beta = reported.beta_new

    # The well-conditioned scalar Fisher direction is 18/(2+penalty).
    # Two halvings provide strict descent for both penalty choices.
    expected = np.full(group_count, 4.5 / (2.0 + penalty))
    allowance = 128 * np.finfo(float).eps
    np.testing.assert_allclose(beta, expected, rtol=allowance, atol=allowance)
    normalized_objective = np.sum((10.0 - np.exp(beta)) ** 2 + 0.5 * penalty * beta**2)
    before = 81.0 * group_count
    assert normalized_objective < before
    assert reported.objective_after / scale == pytest.approx(
        normalized_objective, rel=allowance, abs=allowance * before
    )


@pytest.mark.parametrize("scale", [1e-24, 1.0, 1e24])
@pytest.mark.parametrize("joint", [False, True])
def test_exhausted_scop_line_search_retains_the_committed_state(scale, joint):
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    design = np.array([[-1.0], [1.0]])
    if joint:
        result = scop_joint_newton_step(
            {
                0: {
                    "B_scop": design,
                    "S_scop": np.zeros((1, 1)),
                    "beta_scop": np.zeros(1),
                    "reparam": reparam,
                    "bin_idx": None,
                }
            },
            np.full(2, scale),
            np.array([-10.0, 10.0]),
            0.0,
            [GroupSlice("shape", 0, 1)],
            max_halving=0,
        )[0]
    else:
        result = scop_newton_step(
            design,
            np.full(2, scale),
            np.array([-10.0, 10.0]),
            np.zeros(1),
            reparam,
            np.zeros((1, 1)),
            0.0,
            max_halving=0,
        )
    np.testing.assert_array_equal(result.beta_new, np.zeros(1))
    assert result.step_norm == 0.0
    assert result.objective_after == result.objective_before


@pytest.mark.parametrize("scale", [1e-24, 1.0, 1e24])
@pytest.mark.parametrize("force_uphill_inner", [False, True])
def test_actual_scop_outer_boundary_uses_weighted_objective_units(
    monkeypatch, scale, force_uphill_inner
):
    import superglm.solvers.irls_direct as irls_direct

    if force_uphill_inner:

        def propose_uphill(**kwargs):
            return SCOPNewtonResult(
                beta_new=np.array([9.0]),
                objective_before=81.0 * scale,
                objective_after=scale * (10.0 - np.exp(9.0)) ** 2,
                step_norm=9.0,
                used_fisher_fallback=True,
            )

        monkeypatch.setattr(irls_direct, "scop_newton_step", propose_uphill)
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
        _scop_joint=False,
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
    assert result.iteration_log[0].step_halvings == (2 if force_uphill_inner else 0)


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
