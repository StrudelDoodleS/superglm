"""Pure convergence decisions shared by exact and discrete REML."""

from __future__ import annotations

import numpy as np

from superglm.reml.convergence import (
    evaluate_reml_candidate,
    project_reml_gradient,
)


def test_project_reml_gradient_respects_fixed_and_active_bounds() -> None:
    gradient = np.array([-3.0, 4.0, 5.0, 6.0, -7.0])
    rho = np.array([10.0, -10.0, 0.0, 10.0, -10.0])
    estimated = np.array([True, True, False, True, True])

    projected = project_reml_gradient(
        gradient,
        rho,
        estimated,
        log_lower=-10.0,
        log_upper=10.0,
    )

    np.testing.assert_array_equal(projected, np.array([0.0, 0.0, 0.0, 6.0, -7.0]))
    np.testing.assert_array_equal(gradient, np.array([-3.0, 4.0, 5.0, 6.0, -7.0]))


def test_project_reml_gradient_rejects_misaligned_inputs() -> None:
    with np.testing.assert_raises_regex(ValueError, "identical shapes"):
        project_reml_gradient(
            np.ones(2),
            np.ones(3),
            np.ones(2, dtype=bool),
            log_lower=-10.0,
            log_upper=10.0,
        )


def test_evaluated_candidate_requires_two_evaluations() -> None:
    first = evaluate_reml_candidate(
        iteration=0,
        objective=12.0,
        previous_objective=12.0,
        projected_gradient=np.zeros(2),
        tolerance=1.0,
    )
    second = evaluate_reml_candidate(
        iteration=1,
        objective=12.0,
        previous_objective=12.0,
        projected_gradient=np.zeros(2),
        tolerance=1.0,
    )

    assert not first.converged
    assert np.isinf(first.objective_change)
    assert second.converged
    assert second.objective_change == 0.0
    assert second.projected_gradient_norm == 0.0
    assert second.score_scale == 13.0


def test_evaluated_candidate_requires_both_score_and_objective_tolerance() -> None:
    score_failure = evaluate_reml_candidate(
        iteration=2,
        objective=9.0,
        previous_objective=9.0,
        projected_gradient=np.array([2.0]),
        tolerance=0.1,
    )
    objective_failure = evaluate_reml_candidate(
        iteration=2,
        objective=9.0,
        previous_objective=7.0,
        projected_gradient=np.zeros(1),
        tolerance=0.1,
    )

    assert not score_failure.converged
    assert not objective_failure.converged
