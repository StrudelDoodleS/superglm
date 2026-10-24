"""Penalty-coordinate oracles for SCOP REML."""

from fractions import Fraction
from types import SimpleNamespace

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
            "reparam": SimpleNamespace(free_dim=0),
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
            "reparam": SimpleNamespace(free_dim=0),
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


def test_penalty_quadratic_has_no_cancellation_at_a_large_lambda():
    """The latent penalty is summed with the ordinary one, not left over from a difference.

    In units of 1e9 a SCOP lambda reaches 1e20 with increments near 1e8, where
    the mapped block ``lambda gamma' S gamma`` is some 1e16 times the latent
    penalty: taken away from the whole quadratic, its rounding swamped the
    result and the ordinary penalty with it.  Each term is a nonnegative
    form, so the computed sum is within ``gamma_n`` of the exact one relative
    to the forms' absolute values.
    """
    from superglm._roundoff import gamma
    from superglm.reml.scop_efs import compute_scop_aware_penalty_quad

    ordinary_beta = np.array([1.0, -2.0])
    ordinary_omega = np.array([[2.0, -1.0], [-1.0, 2.0]])
    difference = np.array([[1.0, -2.0, 1.0]])
    latent_omega = difference.T @ difference
    beta_eff = np.log(1e8) + np.array([0.0, 0.01, 0.03])
    lambdas = {"plain": 3e11, "mono": 1e20}
    penalty = np.zeros((5, 5))
    penalty[:2, :2] = lambdas["plain"] * ordinary_omega
    penalty[2:, 2:] = lambdas["mono"] * latent_omega
    states = {
        1: {
            "group_name": "mono",
            "group_sl": slice(2, 5),
            "beta_eff": beta_eff,
            "S_scop": latent_omega,
            "reparam": SimpleNamespace(free_dim=0),
        }
    }

    actual = compute_scop_aware_penalty_quad(
        np.concatenate([ordinary_beta, np.exp(beta_eff)]), penalty, states, lambdas
    )

    def form(matrix, vector, scale=1.0):
        matrix = [[Fraction(value) for value in row] for row in matrix]
        vector = [Fraction(value) for value in vector]
        return Fraction(scale) * sum(
            vector[i] * matrix[i][j] * vector[j]
            for i in range(len(vector))
            for j in range(len(vector))
        )

    exact = form(penalty[:2, :2], ordinary_beta) + form(latent_omega, beta_eff, lambdas["mono"])
    magnitude = float(
        form(np.abs(penalty[:2, :2]), np.abs(ordinary_beta))
        + form(np.abs(latent_omega), np.abs(beta_eff), lambdas["mono"])
    )
    assert abs(Fraction(actual) - exact) <= Fraction(gamma(2 * 5 + 2 * 3 + 4) * magnitude)
