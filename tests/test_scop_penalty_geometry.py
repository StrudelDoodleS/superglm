"""Penalty-coordinate oracles for SCOP REML."""

import numpy as np
import pytest

from superglm.types import PenaltyComponent


def test_efs_matches_scop_component_by_parent_group_not_lambda_name():
    """Named penalty components on a SCOP block must use latent beta_eff."""
    from superglm.reml.scop_efs import _joint_efs_lambda_step

    omega = np.eye(2)
    component = PenaltyComponent(
        name="mono:wiggle",
        group_name="mono",
        group_index=0,
        group_sl=slice(0, 2),
        omega_raw=omega,
        omega_ssp=omega,
        rank=2.0,
    )
    beta_eff = np.array([0.4, -0.2])
    mapped_beta = np.exp(beta_eff)
    states = {
        0: {
            "group_name": "mono",
            "group_sl": slice(0, 2),
            "beta_eff": beta_eff,
        }
    }
    updated, _, _ = _joint_efs_lambda_step(
        [component],
        mapped_beta,
        np.zeros((2, 2)),
        1.0,
        {"mono:wiggle": 1.0},
        {"mono:wiggle"},
        states,
        {"mono:wiggle": 1.0},
        {},
    )

    expected = 2.0 / float(beta_eff @ beta_eff)
    assert abs(np.log(expected)) < 4.0
    assert updated["mono:wiggle"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_efs_step_handles_an_identity_component_that_stores_no_matrix():
    """The unit-level regression for the defect this file's other cases cannot see.

    Every other component here is ``penalty_kind="dense"`` and carries a real
    ``omega_ssp``. A ``RandomEffect`` is ``penalty_kind="identity"`` and stores
    **neither** ``omega_ssp`` nor ``omega_raw`` -- its penalty *is* the identity,
    recorded in the kind rather than as a matrix. Reading the attribute directly
    therefore yielded ``None`` into ``beta @ omega @ beta``.

    Verified by mutation: restoring the raw ``pc.omega_ssp`` read makes this
    raise ``ValueError: matmul: Input operand 1 does not have enough dimensions
    (has 0 ...)`` immediately, rather than surfacing as a slow non-convergence
    several layers up -- which is why it belongs here and not only in an
    end-to-end fit.

    The expected value is the Fellner-Schall ratio with ``Omega = I``: the
    quadratic collapses to ``beta @ beta`` and the trace to ``trace(H_inv)``.
    """
    from superglm.reml.scop_efs import _joint_efs_lambda_step

    component = PenaltyComponent(
        name="grp",
        group_name="grp",
        group_index=0,
        group_sl=slice(0, 3),
        omega_raw=None,
        omega_ssp=None,
        rank=3.0,
        penalty_kind="identity",
    )
    beta = np.array([0.5, -0.25, 0.75])
    hessian_inverse = np.diag([0.1, 0.2, 0.3])
    lam = 1.0

    updated, _, _ = _joint_efs_lambda_step(
        [component],
        beta,
        hessian_inverse,
        1.0,
        {"grp": lam},
        {"grp"},
        {},
        {"grp": 1.0},
        {},
    )

    quadratic = float(beta @ beta)
    trace_term = float(np.trace(hessian_inverse))
    residual_edf = component.rank - lam * trace_term
    expected = lam * np.exp(np.log(residual_edf) - np.log(quadratic * lam))
    assert updated["grp"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_penalty_quadratic_uses_every_named_component_in_latent_coordinates():
    """Overlapping SCOP penalties must not fall back to mapped gamma coordinates."""
    from superglm.reml.scop_efs import compute_scop_aware_penalty_quad

    beta_eff = np.array([0.3, -0.5])
    mapped_beta = np.exp(beta_eff)
    omegas = (np.diag([1.0, 0.0]), np.ones((2, 2)))
    names = ("mono:first", "mono:shared")
    lambdas = {"mono:first": 2.0, "mono:shared": 0.7}
    components = [
        PenaltyComponent(
            name=name,
            group_name="mono",
            group_index=0,
            group_sl=slice(0, 2),
            omega_raw=omega,
            omega_ssp=omega,
            rank=1.0,
        )
        for name, omega in zip(names, omegas, strict=True)
    ]
    penalty = sum(lambdas[name] * omega for name, omega in zip(names, omegas, strict=True))
    states = {
        0: {
            "group_name": "mono",
            "group_sl": slice(0, 2),
            "beta_eff": beta_eff,
        }
    }

    actual = compute_scop_aware_penalty_quad(
        mapped_beta,
        penalty,
        states,
        lambdas,
        reml_penalties=components,
    )
    expected = sum(
        lambdas[name] * float(beta_eff @ omega @ beta_eff)
        for name, omega in zip(names, omegas, strict=True)
    )
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
