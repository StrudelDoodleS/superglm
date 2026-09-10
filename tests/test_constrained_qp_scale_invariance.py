"""Equivalent objective and constraint units preserve the QP certificate."""

import numpy as np
import pytest

from superglm._fit_trace import MemoryTraceSink, TraceRun
from superglm.solvers.constrained_qp import _is_feasible, solve_constrained_qp


@pytest.mark.parametrize("objective_scale", [1e-20, 1e-14, 1.0, 1e14, 1e20])
@pytest.mark.parametrize("row_scale", [1e-14, 1.0, 1e14])
@pytest.mark.parametrize("active", [None, [0], [0, 1]])
def test_scaled_primal_dual_problem_has_the_same_unique_solution(
    objective_scale, row_scale, active
):
    result = solve_constrained_qp(
        objective_scale * np.eye(2),
        objective_scale * np.array([1.0, -1.0]),
        row_scale * np.eye(2),
        np.zeros(2),
        active_set_init=active,
    )

    assert result.converged
    allowance = 64 * np.finfo(float).eps
    np.testing.assert_allclose(result.beta, [1.0, 0.0], rtol=allowance, atol=allowance)
    # Divide out the declared objective/row units before assessing KKT.
    gradient = result.beta - np.array([1.0, -1.0])
    assert np.min(result.beta) >= -allowance
    assert np.min(gradient) >= -allowance
    np.testing.assert_allclose(result.beta * gradient, 0.0, rtol=0.0, atol=allowance)


@pytest.mark.parametrize("row_scale", [1e-20, 1e-14, 1.0, 1e14, 1e20])
def test_small_constraint_row_cannot_certify_a_unit_primal_violation(row_scale):
    A = np.array([[row_scale]])
    assert not _is_feasible(A, np.array([-1.0]), np.zeros(1), 1e-12)
    result = solve_constrained_qp(np.ones((1, 1)), np.array([-1.0]), A, np.zeros(1))
    assert result.converged
    np.testing.assert_allclose(result.beta, [0.0], rtol=0.0, atol=64 * np.finfo(float).eps)


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_scaled_qp_preserves_infeasibility_and_unbounded_refusals(scale):
    result = solve_constrained_qp(
        np.ones((1, 1)),
        np.zeros(1),
        scale * np.array([[1.0], [-1.0]]),
        scale * np.array([1.0, 0.0]),
    )
    assert not result.converged
    with pytest.raises(ValueError, match="null\\(H\\)"):
        solve_constrained_qp(
            scale * np.diag([1.0, 0.0]), scale * np.ones(2), np.empty((0, 2)), np.empty(0)
        )
    with pytest.raises(ValueError, match="PSD|indefinite|negative"):
        solve_constrained_qp(
            scale * np.diag([1.0, -1.0]), scale * np.ones(2), np.empty((0, 2)), np.empty(0)
        )


def test_zero_constraint_action_is_exactly_feasible():
    assert _is_feasible(np.zeros((1, 2)), np.zeros(2), np.zeros(1), 1e-12)
    assert not _is_feasible(np.zeros((1, 2)), np.zeros(2), np.ones(1), 1e-12)


@pytest.mark.parametrize("row_scale", [1e-300, 1.0, 1e300])
@pytest.mark.parametrize("active", [None, [0]])
def test_nonzero_constraint_action_cannot_underflow_into_structural_zero(row_scale, active):
    A = np.array([[row_scale]])
    g = np.array([-1e-30])
    assert not _is_feasible(A, g, np.zeros(1), 1e-12)
    result = solve_constrained_qp(np.ones((1, 1)), g, A, np.zeros(1), active_set_init=active)
    assert result.converged
    np.testing.assert_array_equal(result.beta, np.zeros(1))


def test_integer_constraint_rows_preserve_the_same_primal_dual_problem():
    result = solve_constrained_qp(
        np.eye(2),
        np.array([1.0, -1.0]),
        np.eye(2, dtype=int),
        np.zeros(2, dtype=int),
        active_set_init=[0, 1],
    )
    assert result.converged
    np.testing.assert_array_equal(result.beta, [1.0, 0.0])


def test_blocking_trace_uses_the_same_range_safe_direction_certificate():
    sink = MemoryTraceSink()
    result = solve_constrained_qp(
        np.ones((1, 1)),
        np.array([-1e-30]),
        np.array([[1e-300]]),
        np.zeros(1),
        _trace_run=TraceRun("small-action", sink=sink),
    )
    assert result.converged
    decisions = [event.payload for event in sink.events]
    assert len(decisions) == 1
    assert decisions[0]["blocking_row"] == 0
    assert decisions[0]["blocking_is_considered"]
    assert decisions[0]["blocking_scaled_step"] == -1.0


@pytest.mark.parametrize("objective_scale", [1e-14, 1.0, 1e14])
@pytest.mark.parametrize("row_scale", [1e-14, 1.0, 1e14])
@pytest.mark.parametrize("redundant", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_identified_homogeneous_active_face_has_its_exact_zero_representative(
    objective_scale, row_scale, redundant, warm
):
    # The independent rows identify the face {0}. Positive multipliers
    # certify that this face is the unique optimum for this positive H.
    A = np.array([[1.0, 1.0], [1.0, 2.0]])
    if redundant:
        A = np.vstack((A, A[0]))
    force = A.T @ np.arange(1.0, len(A) + 1.0)
    result = solve_constrained_qp(
        objective_scale * np.array([[2.0, 1.0], [1.0, 3.0]]),
        -objective_scale * force,
        row_scale * A,
        np.zeros(len(A)),
        active_set_init=list(range(len(A))) if warm else None,
    )

    assert result.converged
    np.testing.assert_array_equal(result.beta, np.zeros(2))
    assert result.rank == result.width == 2
    assert _is_feasible(row_scale * A, result.beta, np.zeros(len(A)), 1e-12)


def test_homogeneous_active_face_preserves_a_nonzero_tangent_solution():
    result = solve_constrained_qp(
        np.eye(2),
        np.array([-1.0, 2.0]),
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        np.zeros(2),
        active_set_init=[0, 1],
    )

    assert result.converged
    np.testing.assert_allclose(result.beta, [0.0, 2.0], rtol=0.0, atol=64 * np.finfo(float).eps)


def test_full_rank_affine_active_face_preserves_its_nonzero_representative():
    result = solve_constrained_qp(
        np.eye(2),
        np.zeros(2),
        np.eye(2),
        np.array([1.0, 2.0]),
        active_set_init=[0, 1],
    )

    assert result.converged
    np.testing.assert_array_equal(result.beta, [1.0, 2.0])


def test_wrong_full_rank_homogeneous_warm_face_releases_its_constraint():
    result = solve_constrained_qp(
        np.eye(2),
        np.array([1.0, -1.0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.zeros(2),
        active_set_init=[0, 1],
    )

    assert result.converged
    np.testing.assert_array_equal(result.beta, [1.0, 0.0])
