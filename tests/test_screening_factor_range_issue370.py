"""Weighted numeric screening designs can fit inside float64 when raw shifts do not."""

import numpy as np
import pytest

import superglm.screening._pair_factor as pair_factor
from superglm.screening._pair_factor import PairFactor, numeric_cat_factor
from superglm.screening._score_stat import penalized_score_statistic_ladder


def _weighted_row_reference(codes, menu, z, score, weights):
    """QR the actual weighted joint rows without taking any raw differences."""
    root = np.sqrt(weights)
    weighted_z = root * z
    response = np.divide(score, root, out=np.zeros_like(score), where=root > 0.0)
    rows = np.column_stack(
        (
            root,
            weighted_z,
            root[:, None] * menu[codes],
            weighted_z[:, None] * menu[codes],
            response,
        )
    )
    reduced = np.linalg.qr(rows, mode="r")
    width = rows.shape[1]
    joint = np.zeros((width, width))
    joint[: reduced.shape[0]] = reduced
    return PairFactor(joint, 2 + menu.shape[1], menu.shape[1]), rows


def _case(n_levels, *, nonzero_score):
    codes = np.repeat(np.arange(n_levels), 2)
    menu = np.array([[0.0], [1.0]]) if n_levels == 2 else np.ones((1, 1))
    z = np.tile([-1e308, 1e308], n_levels)
    weights = np.full(z.size, 4e-309)
    response = np.array([-0.8, 0.3, 0.7, -0.5])[: z.size]
    score = np.sqrt(weights) * response if nonzero_score else np.zeros_like(z)
    return codes, n_levels, menu, z, score, weights


@pytest.mark.parametrize("n_levels", [1, 2])
@pytest.mark.parametrize("nonzero_score", [False, True])
def test_a_finite_weighted_design_survives_an_overflowing_raw_level_shift(n_levels, nonzero_score):
    """Removing the weighted-row fallback must recover issue #370's overflow."""
    codes, n_levels, menu, z, score, weights = _case(n_levels, nonzero_score=nonzero_score)
    reference, rows = _weighted_row_reference(codes, menu, z, score, weights)
    assert np.isfinite(rows).all()
    assert np.isfinite(rows.T @ rows).all()

    with np.errstate(over="raise", invalid="raise"):
        factor = numeric_cat_factor(codes, n_levels, menu, z, score, weights)
        rung = penalized_score_statistic_ladder(factor, None, budgets=(4.0,))[0]

    assert np.isfinite(factor.joint).all()
    # Equilibrate before forming a Gram: one column is about 1e-154, another
    # 1e154.  A single absolute tolerance on their raw Gram would miss either
    # the weighted intercept or the score channel entirely.
    scales = np.max(np.abs(rows), axis=0)
    scales[scales == 0.0] = 1.0
    normalized = factor.joint / scales
    row_normalized = rows / scales
    np.testing.assert_allclose(
        normalized.T @ normalized,
        row_normalized.T @ row_normalized,
        rtol=64 * np.finfo(float).eps,
        atol=64 * np.finfo(float).eps,
    )
    expected = penalized_score_statistic_ladder(reference, None, budgets=(4.0,))[0]
    assert rung.statistic == pytest.approx(expected.statistic, rel=1e-13, abs=1e-14)
    assert rung.edf0 == pytest.approx(expected.edf0, abs=1e-13)
    assert rung.reference_variance == pytest.approx(expected.reference_variance, abs=1e-13)
    # At two levels, the one residual direction is [1, -1, -1, 1] / 2.
    # Its response projection is -1.15, hence a squared score of 1.3225.
    expected_statistic = 1.3225 if nonzero_score and n_levels == 2 else 0.0
    assert rung.statistic == pytest.approx(expected_statistic, rel=1e-13, abs=1e-14)
    assert rung.edf0 == float(n_levels - 1)
    assert rung.reference_variance == float(2 * (n_levels - 1))


def test_the_range_fallback_preserves_response_energy_across_row_chunks(monkeypatch):
    """Dropping a partial level factor must change both its Gram and score."""
    monkeypatch.setattr(pair_factor, "_FACTOR_CHUNK_DOUBLES", 6)
    codes = np.repeat([0, 1], 4)
    menu = np.array([[0.0], [1.0]])
    z = np.tile([-1e308, 1e308], 4)
    weights = np.full(z.size, 1e-309)
    response = np.array([-0.8, 0.3, 0.4, 0.9, 0.7, -0.5, -0.4, 1.1])
    score = np.sqrt(weights) * response
    reference, rows = _weighted_row_reference(codes, menu, z, score, weights)
    with np.errstate(over="raise", invalid="raise"):
        factor = numeric_cat_factor(codes, 2, menu, z, score, weights)
        rung = penalized_score_statistic_ladder(factor, None, budgets=(4.0,))[0]
    scales = np.max(np.abs(rows), axis=0)
    actual = factor.joint / scales
    expected = reference.joint / scales
    np.testing.assert_allclose(
        actual.T @ actual,
        expected.T @ expected,
        rtol=64 * np.finfo(float).eps,
        atol=64 * np.finfo(float).eps,
    )
    assert rung.statistic == pytest.approx(1.3**2 / 8, rel=1e-13)
    assert rung.edf0 == 1.0


@pytest.mark.parametrize("position", [0, 4])
@pytest.mark.parametrize("excluded_z", [-np.finfo(float).max, np.finfo(float).max])
def test_an_excluded_row_does_not_change_the_range_fallback(position, excluded_z):
    """A zero-weight row must stay outside even an affected level's row QR."""
    codes, n_levels, menu, z, score, weights = _case(2, nonzero_score=True)
    with np.errstate(over="raise", invalid="raise"):
        baseline = numeric_cat_factor(codes, n_levels, menu, z, score, weights)
        factor = numeric_cat_factor(
            np.insert(codes, position, 0),
            n_levels,
            menu,
            np.insert(z, position, excluded_z),
            np.insert(score, position, 0.0),
            np.insert(weights, position, 0.0),
        )
    np.testing.assert_array_equal(factor.joint, baseline.joint)
