import math

import pytest
from scipy.stats import chi2

from superglm.profiling._scalar import (
    RecordedObjective,
    likelihood_ratio_interval,
    minimize_profile,
)

N = 1000.0
CURVATURE = 3.0


def quadratic(x):
    return 0.5 * CURVATURE * (x - 0.3) ** 2


def test_minimize_profile_records_bounds_and_finds_minimum():
    objective = RecordedObjective(quadratic)
    assert minimize_profile(objective, (0.0, 1.0), xatol=1e-6, maxiter=50)
    x_hat, _ = objective.best()
    assert x_hat == pytest.approx(0.3, abs=1e-5)
    assert 0.0 in objective.values and 1.0 in objective.values


def test_boundary_minimum_is_the_recorded_bound():
    objective = RecordedObjective(lambda x: x)
    minimize_profile(objective, (0.2, 1.0), xatol=1e-6, maxiter=50)
    assert objective.best()[0] == 0.2


# An infinite value inside Brent's parabolic step forms inf - inf and warns.
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_infeasible_points_are_routed_around_and_never_best():
    objective = RecordedObjective(lambda x: math.inf if x > 0.6 else quadratic(x))
    assert minimize_profile(objective, (0.0, 1.0), xatol=1e-6, maxiter=50)
    assert math.isinf(objective.values[1.0])
    assert objective.best()[0] == pytest.approx(0.3, abs=1e-5)


def test_interval_is_exact_for_a_quadratic_profile():
    half_width = math.sqrt(chi2.ppf(0.95, 1) / (N * CURVATURE))
    interval, _, _ = likelihood_ratio_interval(
        RecordedObjective(quadratic), 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=N, xtol=1e-12
    )
    assert interval.lower == pytest.approx(0.3 - half_width, rel=1e-9)
    assert interval.upper == pytest.approx(0.3 + half_width, rel=1e-9)
    assert not (interval.lower_censored or interval.upper_censored)


def test_interval_after_a_search_costs_a_few_evaluations_per_side():
    # Every evaluation is a model fit. Seeded from the searched points, each
    # side of a quadratic profile is bracketed at its first point and then
    # needs brentq's secant steps only; locating a side from its bound, as the
    # first implementation did, took 11 fits per side here.
    objective = RecordedObjective(quadratic)
    minimize_profile(objective, (0.0, 1.0), xatol=1e-3, maxiter=50)
    x_hat, nll_hat = objective.best()
    searched = len(objective.values)
    interval, _, _ = likelihood_ratio_interval(
        objective, x_hat, nll_hat, (0.0, 1.0), alpha=0.05, scale=N, xtol=1e-4
    )
    half_width = math.sqrt(chi2.ppf(0.95, 1) / (N * CURVATURE))
    assert interval.lower == pytest.approx(0.3 - half_width, abs=1e-4)
    assert interval.upper == pytest.approx(0.3 + half_width, abs=1e-4)
    assert len(objective.values) - searched <= 8


def test_side_inside_the_acceptance_region_is_censored_at_the_bound():
    interval, _, _ = likelihood_ratio_interval(
        RecordedObjective(quadratic), 0.3, 0.0, (0.29, 0.31), alpha=0.05, scale=N, xtol=1e-12
    )
    assert (interval.lower, interval.upper) == (0.29, 0.31)
    assert interval.lower_censored and interval.upper_censored


def test_interval_side_ending_at_infeasible_region_is_censored():
    def objective(x):
        return math.inf if x > 0.32 else quadratic(x)

    interval, _, _ = likelihood_ratio_interval(
        RecordedObjective(objective), 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=N, xtol=1e-12
    )
    assert interval.upper_censored and interval.upper == pytest.approx(0.32, abs=1e-9)
    assert not interval.lower_censored


def test_a_crossing_just_before_an_infeasible_region_is_not_censored():
    # The crossing at 0.3 + half_width is genuine even with the wall one step
    # beyond it: censoring follows the root's bracket partner, not the size
    # of the excess at the root.
    half_width = math.sqrt(chi2.ppf(0.95, 1) / (N * CURVATURE))

    def objective(x):
        return math.inf if x > 0.3 + 1.001 * half_width else quadratic(x)

    interval, _, _ = likelihood_ratio_interval(
        RecordedObjective(objective), 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=N, xtol=1e-12
    )
    assert interval.upper == pytest.approx(0.3 + half_width, rel=1e-9)
    assert not interval.upper_censored


def test_a_level_below_half_an_ulp_of_one_has_a_finite_cutoff():
    # 1 - alpha rounds to 1 here, where chi2.ppf is infinite and every side
    # would read as censored; the upper tail itself is finite.
    alpha = math.ulp(0.0)
    cutoff = chi2.isf(alpha, 1)
    assert math.isfinite(cutoff)
    interval, _, _ = likelihood_ratio_interval(
        RecordedObjective(quadratic), 0.3, 0.0, (-5.0, 5.0), alpha=alpha, scale=N, xtol=1e-12
    )
    half_width = math.sqrt(cutoff / (N * CURVATURE))
    assert not (interval.lower_censored or interval.upper_censored)
    assert interval.lower == pytest.approx(0.3 - half_width, rel=1e-9)
    assert interval.upper == pytest.approx(0.3 + half_width, rel=1e-9)


def test_an_interval_that_finds_a_deeper_well_is_inverted_from_it():
    # x_hat = 0.3 is a local minimum; the well at 0.5 is one unit deeper, and the
    # rise between them stays under the cutoff, so a side's evaluations reach it.
    def two_wells(x):
        return min(40.0 * (x - 0.3) ** 2, 40.0 * (x - 0.5) ** 2 - 1.0)

    objective = RecordedObjective(two_wells)
    interval, centre, centre_nll = likelihood_ratio_interval(
        objective, 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=1.0, xtol=1e-12
    )
    assert centre_nll < 0.0 and (centre, centre_nll) == objective.best()
    # Each end is a crossing measured from that deeper point, not from x_hat.
    cutoff = chi2.isf(0.05, 1)
    for end in (interval.lower, interval.upper):
        assert 2.0 * (two_wells(end) - centre_nll) == pytest.approx(cutoff, abs=1e-8)
    assert not (interval.lower_censored or interval.upper_censored)
