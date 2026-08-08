"""Pure convergence decisions shared by exact and discrete REML."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.reml.convergence import (
    FLAT_DIRECTION_CURVATURE_ANCHOR,
    FLAT_DIRECTION_CURVATURE_REL,
    FLAT_DIRECTION_FREEZE_FLOOR,
    classify_dead_feasible_exit,
    evaluate_reml_candidate,
    freeze_flat_directions,
    mask_frozen_stop_gradient,
    project_reml_gradient,
    trial_counts_as_precision_evidence,
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
    The curvature arm therefore judges each direction's row curvature
    against the strongest estimated direction, not against the objective's
    magnitude."""
    tiny = np.full(3, 1e-9)
    hess = np.diag([4.0, 0.03, 0.5])
    estimated = np.ones(3, dtype=bool)

    decision = freeze_flat_directions(
        tiny, hess, np.ones(3), estimated, objective=1e7, tolerance=1e-6
    )

    # Bar = REL * max(|H_jj|) = 0.01 * 4.0: only the 0.03 direction is flat.
    # Under the old absolute bar (1e-7 * 1e7 = 1.0) the 0.5 direction froze.
    np.testing.assert_array_equal(decision.frozen, np.array([False, True, False]))
    assert decision.curvature_bar == pytest.approx(0.04)

    # The verdict is invariant to the objective's magnitude.
    small_objective = freeze_flat_directions(
        tiny, hess, np.ones(3), estimated, objective=1e3, tolerance=1e-6
    )
    np.testing.assert_array_equal(decision.frozen, small_objective.frozen)


def test_freeze_normalizes_curvature_by_penalty_rank() -> None:
    """Row curvature scales with penalty rank: a 600-level random effect
    measures ~255 while an informative low-rank spline measures ~2.5, and
    the raw ratio-to-strongest bar (1e-2 * 255 = 2.55) froze the spline
    with real signal (measured on the mixed-rank fixture). Per rank the
    same directions measure ~0.43 and ~0.82 -- commensurate -- so the
    curvature arm compares curvature per penalty dimension."""
    tiny = np.full(2, 1e-9)
    hess = np.diag([255.0, 2.45])
    ranks = np.array([600.0, 3.0])

    decision = freeze_flat_directions(
        tiny, hess, ranks, np.ones(2, dtype=bool), objective=1e5, tolerance=1e-6
    )

    np.testing.assert_array_equal(decision.frozen, np.array([False, False]))
    np.testing.assert_allclose(decision.row_curvature, [255.0, 2.45])


def test_freeze_normalizes_coupled_curvature_symmetrically() -> None:
    """A shared cross-term belongs to BOTH directions: dividing each row
    by its own rank alone reads H=[[0, .5], [.5, 0]] with ranks [1000, 1]
    as per-rank [5e-4, 0.5] -- the high-rank half freezes, and the
    reduced Newton solve on its orphaned partner sees a zero
    one-dimensional Hessian where the only curvature was joint. The
    matrix is scaled symmetrically (H_ij / sqrt(r_i * r_j), diagonals
    unchanged), so a coupled pair freezes together or stays together."""
    tiny = np.full(2, 1e-9)
    coupled = np.array([[0.0, 0.5], [0.5, 0.0]])
    ranks = np.array([1000.0, 1.0])

    decision = freeze_flat_directions(
        tiny, coupled, ranks, np.ones(2, dtype=bool), objective=1e5, tolerance=1e-6
    )

    np.testing.assert_array_equal(decision.frozen, np.array([False, False]))
    # Both halves judge the shared term at the same normalized value.
    np.testing.assert_allclose(
        decision.normalized_curvature, [0.5 / np.sqrt(1000.0), 0.5 / np.sqrt(1000.0)]
    )

    # A genuinely flat coupled pair freezes together, not asymmetrically.
    weak = freeze_flat_directions(
        tiny,
        np.array([[0.0, 3e-3], [3e-3, 0.0]]),
        ranks,
        np.ones(2, dtype=bool),
        objective=1e5,
        tolerance=1e-6,
    )
    np.testing.assert_array_equal(weak.frozen, np.array([True, True]))


def test_freeze_sees_coupled_curvature_not_just_the_diagonal() -> None:
    """REML Hessians carry cross-terms (multi-penalty anisotropy adds them
    explicitly) and need not be positive definite: a coordinate can hold a
    small diagonal with large off-diagonal curvature. A [[0, c], [c, 0]]
    block has zero diagonals yet real curvature in the coupled
    eigenvector -- judged by the diagonal alone, both directions freeze at
    tiny gradients and the pair is never solved. The curvature arm judges
    each direction's row over the estimated block."""
    tiny = np.full(2, 1e-9)
    coupled = np.array([[0.0, 0.5], [0.5, 0.0]])

    decision = freeze_flat_directions(
        tiny, coupled, np.ones(2), np.ones(2, dtype=bool), objective=1e3, tolerance=1e-6
    )

    np.testing.assert_array_equal(decision.frozen, np.array([False, False]))

    # Coupling to a FIXED direction does not keep a direction alive: the
    # fixed lambda never moves, so that cross-curvature is not exploitable.
    fixed_partner = freeze_flat_directions(
        tiny,
        np.array([[5e-4, 0.5], [0.5, 4.0]]),
        np.ones(2),
        np.array([True, False]),
        objective=1e3,
        tolerance=1e-6,
    )
    np.testing.assert_array_equal(fixed_partner.frozen, np.array([True, True]))


def test_freeze_anchors_the_curvature_scale_when_every_direction_is_weak() -> None:
    """An all-null model has no strong direction to anchor the ratio; the
    absolute anchor bounds its march instead of letting the last null
    direction chase the lambda cap forever. In per-rank units the
    calibrated span is: fully null directions measure <= 6e-5, the
    tightest informative direction 5.2e-3, so the anchored bar
    1e-2 * 0.1 = 1e-3 splits them."""
    assert FLAT_DIRECTION_CURVATURE_ANCHOR == 0.1
    tiny = np.full(2, 1e-9)
    hess = np.diag([4e-4, 2e-4])

    decision = freeze_flat_directions(
        tiny, hess, np.ones(2), np.ones(2, dtype=bool), objective=1e3, tolerance=1e-6
    )

    np.testing.assert_array_equal(decision.frozen, np.array([True, True]))


def test_freeze_requires_the_gradient_arm_and_fixed_lambdas_stay_frozen() -> None:
    gradient = np.array([5.0, 1e-9, 1e-9])
    hess = np.diag([4e-4, 2e-4, 3.0])
    estimated = np.array([True, True, False])

    decision = freeze_flat_directions(
        gradient, hess, np.ones(3), estimated, objective=1e3, tolerance=1e-6
    )

    # A live gradient keeps a weak-curvature direction active; a fixed
    # lambda is frozen whatever its curvature.
    np.testing.assert_array_equal(decision.frozen, np.array([False, True, True]))


def test_freeze_gradient_arm_still_couples_to_a_loose_tolerance() -> None:
    assert FLAT_DIRECTION_FREEZE_FLOOR == 1e-7
    assert FLAT_DIRECTION_CURVATURE_REL == 1e-2
    gradient = np.array([0.5])
    hess = np.diag([2e-4])
    estimated = np.ones(1, dtype=bool)

    tight = freeze_flat_directions(
        gradient, hess, np.ones(1), estimated, objective=1e3, tolerance=1e-6
    )
    loose = freeze_flat_directions(
        gradient, hess, np.ones(1), estimated, objective=1e3, tolerance=1e-2
    )

    # freeze_tol = max(0.1*tol, floor): 0.5 is live at 1e-4*scale but
    # under the loose 1e-3*scale bar.
    np.testing.assert_array_equal(tight.frozen, np.array([False]))
    np.testing.assert_array_equal(loose.frozen, np.array([True]))


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


def test_precision_evidence_needs_a_converged_finite_trial() -> None:
    """On the Fisher path an exhausted-PIRLS trial is still scored; its
    objective sits at a non-stationary beta, so an Armijo rejection of it
    proves nothing about the true profile objective. Only a converged
    trial with a finite objective is evidence for the precision exit --
    the same standard the observed-geometry path enforces by skipping
    unconverged trials before evaluation."""
    assert trial_counts_as_precision_evidence(True, 12.5)
    assert not trial_counts_as_precision_evidence(False, 12.5)
    assert not trial_counts_as_precision_evidence(True, float("nan"))
    assert not trial_counts_as_precision_evidence(True, float("inf"))


def test_dead_feasible_exit_needs_an_evaluated_trial() -> None:
    """had_feasible_trial only means a rho proposal differed. On the
    observed-geometry path every trial can be SKIPPED -- PIRLS did not
    converge, geometry failed, the mode did not certify -- without any
    objective evaluated. A dead search that evaluated nothing is no proof
    the optimum is resolved; it stays an honest failure."""
    starved = classify_dead_feasible_exit(
        1e-9, objective=1e4, tolerance=1e-6, evaluated_trial=False
    )
    evidenced = classify_dead_feasible_exit(
        1e-9, objective=1e4, tolerance=1e-6, evaluated_trial=True
    )

    assert starved == "line_search_failed"
    assert evidenced == "converged_at_precision"


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
