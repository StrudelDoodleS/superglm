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
    interval = likelihood_ratio_interval(
        quadratic, 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=N, rtol=1e-12
    )
    assert interval.lower == pytest.approx(0.3 - half_width, rel=1e-9)
    assert interval.upper == pytest.approx(0.3 + half_width, rel=1e-9)
    assert not (interval.lower_censored or interval.upper_censored)


def test_side_inside_the_acceptance_region_is_censored_at_the_bound():
    interval = likelihood_ratio_interval(
        quadratic, 0.3, 0.0, (0.29, 0.31), alpha=0.05, scale=N, rtol=1e-12
    )
    assert (interval.lower, interval.upper) == (0.29, 0.31)
    assert interval.lower_censored and interval.upper_censored


def test_interval_side_ending_at_infeasible_region_is_censored():
    def objective(x):
        return math.inf if x > 0.32 else quadratic(x)

    interval = likelihood_ratio_interval(
        objective, 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=N, rtol=1e-12
    )
    assert interval.upper_censored and interval.upper == pytest.approx(0.32, abs=1e-9)
    assert not interval.lower_censored
