"""Range regressions for public consumers of binary64 numerical kernels."""

import numpy as np
import pytest


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_constraint_feasibility_retains_dimensionless_extreme_actions(monkeypatch, scale):
    from superglm.solvers.constrained_qp import _is_feasible

    monkeypatch.setattr(np, "longdouble", np.float64)
    beta = np.array([-1e-30 if scale < 1 else -1e30])
    assert not _is_feasible(np.array([[scale]]), beta, np.zeros(1), 1e-12)


def test_centered_moments_do_not_overflow_the_outer_product(monkeypatch):
    from superglm.inference._metrics_design import centered_gram_from_moments

    monkeypatch.setattr(np, "longdouble", np.float64)
    result = centered_gram_from_moments(np.array([[4e200]]), np.array([1e200]), 1e200)
    np.testing.assert_allclose(result, [[3e200]], rtol=4 * np.finfo(float).eps, atol=0)


def test_weighted_mean_retains_representable_extreme_contribution(monkeypatch):
    from superglm.validation import _weighted_mean

    monkeypatch.setattr(np, "longdouble", np.float64)
    maximum, smallest = np.finfo(float).max, np.nextafter(0.0, 1.0)
    # The maximum weight sits on a zero value, so only the smallest weight's
    # product enters the numerator and the column ranges stay representable.
    assert (
        _weighted_mean(np.array([0.0, maximum]), np.array([maximum, smallest]), "test") == smallest
    )


def test_scop_roundoff_scales_before_squaring_large_actions(monkeypatch):
    from decimal import Decimal

    from superglm.solvers.scop_newton import _objective_delta_roundoff

    def bound(scale):
        return _objective_delta_roundoff(
            gammas=[np.array([scale])],
            trial_gammas=[np.array([2 * scale])],
            betas=[np.zeros(1)],
            trial_betas=[np.zeros(1)],
            grams=[np.eye(1)],
            penalties=[np.zeros((1, 1))],
            lambdas=[0.0],
            response_norm=scale,
            n_rows=2,
        )

    ordinary = bound(1.0)
    monkeypatch.setattr(np, "longdouble", np.float64)
    actual = bound(1e160)
    expected = float(Decimal.from_float(ordinary) * Decimal.from_float(1e160) ** 2)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=32 * np.finfo(float).eps)


def test_penalty_roundoff_retains_balanced_extreme_coordinate_products():
    from superglm.solvers.scop_newton import _positive_quadratic_roundoff

    actual = _positive_quadratic_roundoff(
        np.array([1e150, 1e-150]),
        np.diag([1e-300, 1e300]),
        1e-14,
    )
    assert actual == pytest.approx(2e-14, rel=16 * np.finfo(float).eps, abs=0)


def test_scop_roundoff_retains_small_action_times_large_response():
    from superglm.solvers.scop_newton import _objective_delta_roundoff

    actual = _objective_delta_roundoff(
        gammas=[np.array([1e-300])],
        trial_gammas=[np.array([2e-300])],
        betas=[np.zeros(1)],
        trial_betas=[np.zeros(1)],
        grams=[np.eye(1)],
        penalties=[np.zeros((1, 1))],
        lambdas=[0.0],
        response_norm=1e300,
        n_rows=2,
    )
    unit = np.finfo(float).eps / 2
    gamma = (50 * unit) / (1 - 50 * unit)
    expected = 4 * gamma / (1 - gamma) ** 2.5
    assert actual == pytest.approx(expected, rel=32 * np.finfo(float).eps, abs=0)
