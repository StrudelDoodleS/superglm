"""Pure convergence decisions shared by exact and discrete REML."""

from __future__ import annotations

import numpy as np

from superglm.reml.convergence import (
    FLAT_DIRECTION_CURVATURE_ANCHOR,
    FLAT_DIRECTION_CURVATURE_REL,
    FLAT_DIRECTION_FREEZE_FLOOR,
    classify_dead_feasible_exit,
    evaluate_reml_candidate,
    freeze_flat_directions,
    mask_frozen_stop_gradient,
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


def test_freeze_judges_curvature_relative_to_the_strongest_direction() -> None:
    """score_scale = 1+|objective| grows with n while log-lambda curvature
    saturates, so an absolute curvature bar freezes informative directions
    at large n (measured: the flat-lambda stress design froze f7 at 400k
    rows and everything at 1e6, publishing lambdas a factor e^5.6 off).
    The curvature arm therefore judges each |H_ii| against the strongest
    estimated direction, not against the objective's magnitude."""
    tiny = np.full(3, 1e-9)
    hess = np.array([4.0, 0.03, 0.5])
    estimated = np.ones(3, dtype=bool)

    frozen = freeze_flat_directions(
        tiny, hess, estimated, objective=1e7, tolerance=1e-6
    )

    # Bar = REL * max(|H_jj|) = 0.01 * 4.0: only the 0.03 direction is flat.
    # Under the old absolute bar (1e-7 * 1e7 = 1.0) the 0.5 direction froze.
    np.testing.assert_array_equal(frozen, np.array([False, True, False]))

    # The verdict is invariant to the objective's magnitude.
    small_objective = freeze_flat_directions(
        tiny, hess, estimated, objective=1e3, tolerance=1e-6
    )
    np.testing.assert_array_equal(frozen, small_objective)


def test_freeze_anchors_the_curvature_scale_when_every_direction_is_weak() -> None:
    """An all-null model has no strong direction to anchor the ratio; the
    absolute anchor bounds its march instead of letting the last null
    direction chase the lambda cap forever."""
    assert FLAT_DIRECTION_CURVATURE_ANCHOR == 1.0
    tiny = np.full(2, 1e-9)
    hess = np.array([0.004, 0.002])

    frozen = freeze_flat_directions(
        tiny, hess, np.ones(2, dtype=bool), objective=1e3, tolerance=1e-6
    )

    np.testing.assert_array_equal(frozen, np.array([True, True]))


def test_freeze_requires_the_gradient_arm_and_fixed_lambdas_stay_frozen() -> None:
    gradient = np.array([5.0, 1e-9, 1e-9])
    hess = np.array([0.004, 0.002, 3.0])
    estimated = np.array([True, True, False])

    frozen = freeze_flat_directions(
        gradient, hess, estimated, objective=1e3, tolerance=1e-6
    )

    # A live gradient keeps a weak-curvature direction active; a fixed
    # lambda is frozen whatever its curvature.
    np.testing.assert_array_equal(frozen, np.array([False, True, True]))


def test_freeze_gradient_arm_still_couples_to_a_loose_tolerance() -> None:
    assert FLAT_DIRECTION_FREEZE_FLOOR == 1e-7
    assert FLAT_DIRECTION_CURVATURE_REL == 1e-2
    gradient = np.array([0.5])
    hess = np.array([0.002])
    estimated = np.ones(1, dtype=bool)

    tight = freeze_flat_directions(
        gradient, hess, estimated, objective=1e3, tolerance=1e-6
    )
    loose = freeze_flat_directions(
        gradient, hess, estimated, objective=1e3, tolerance=1e-2
    )

    # freeze_tol = max(0.1*tol, floor): 0.5 is live at 1e-4*scale but
    # under the loose 1e-3*scale bar.
    np.testing.assert_array_equal(tight, np.array([False]))
    np.testing.assert_array_equal(loose, np.array([True]))


def test_stop_mask_passes_a_reactivated_direction_through() -> None:
    """The stop criterion judges iteration k with iteration k-1's freeze
    mask; a frozen direction whose gradient has since grown past the
    freeze bar must not be hidden by that stale mask."""
    previous = np.array([True, False])

    still_flat = mask_frozen_stop_gradient(
        np.array([1e-9, 2.0]), previous, objective=1e3, tolerance=1e-6
    )
    reactivated = mask_frozen_stop_gradient(
        np.array([0.5, 2.0]), previous, objective=1e3, tolerance=1e-6
    )
    unmasked = mask_frozen_stop_gradient(
        np.array([0.5, 2.0]), None, objective=1e3, tolerance=1e-6
    )

    np.testing.assert_array_equal(still_flat, np.array([0.0, 2.0]))
    np.testing.assert_array_equal(reactivated, np.array([0.5, 2.0]))
    np.testing.assert_array_equal(unmasked, np.array([0.5, 2.0]))


def test_dead_feasible_exit_classifies_against_the_resolved_tolerance() -> None:
    """A dead feasible line search at reml_tol=1e-3 with the active
    gradient at 1e-5 of scale has met the precision the caller asked for;
    holding it to the 1e-7 floor misreported it as line_search_failed."""
    loose = classify_dead_feasible_exit(0.1, objective=1e4, tolerance=1e-3)
    tight = classify_dead_feasible_exit(0.1, objective=1e4, tolerance=1e-11)
    tight_resolved = classify_dead_feasible_exit(
        1e-4, objective=1e4, tolerance=1e-11
    )

    assert loose == "converged_at_precision"
    assert tight == "line_search_failed"
    assert tight_resolved == "converged_at_precision"


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
