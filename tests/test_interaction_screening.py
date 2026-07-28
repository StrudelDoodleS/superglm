"""Per-pair screening moments: exactness pins against the dense row-Kronecker.

Task 1 of docs/superpowers/plans/2026-07-28-interaction-screening.md.  The
whole screening design rests on the cell-space assembly reproducing the dense
assembly exactly, so these tolerances must not be loosened.
"""

from __future__ import annotations

import numpy as np

from superglm.distributions import Gamma, Poisson
from superglm.links import LogLink
from superglm.screening import pair_cell_moments, pair_score_curvature, working_score


def _pair_case(seed, n=4000, n_a=17, n_b=13, k_a=4, k_b=3, signed=False):
    rng = np.random.default_rng(seed)
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    B_a = rng.normal(size=(n_a, k_a))
    B_b = rng.normal(size=(n_b, k_b))
    score = rng.normal(size=n) if signed else rng.uniform(0.1, 1.0, n)
    weights = rng.normal(size=n) if signed else rng.uniform(0.2, 2.0, n)
    return codes_a, codes_b, B_a, B_b, score, weights


def _dense_row_kronecker(codes_a, codes_b, B_a, B_b):
    rows_a = B_a[codes_a]
    rows_b = B_b[codes_b]
    return np.einsum("rp,rq->rpq", rows_a, rows_b).reshape(
        len(codes_a), B_a.shape[1] * B_b.shape[1]
    )


def test_cell_assembly_matches_dense_row_kronecker():
    codes_a, codes_b, B_a, B_b, score, weights = _pair_case(0)

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 17, 13, score, weights)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)

    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    U_dense = X.T @ score
    V_dense = X.T @ (X * weights[:, None])
    np.testing.assert_allclose(U, U_dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V, V_dense, rtol=1e-12, atol=1e-12)


def test_cell_assembly_handles_signed_score_and_weights():
    """REML working quantities are signed; nothing here may assume positivity."""
    codes_a, codes_b, B_a, B_b, score, weights = _pair_case(1, signed=True)

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 17, 13, score, weights)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)

    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(V, X.T @ (X * weights[:, None]), rtol=1e-11, atol=1e-11)


def test_empty_and_singleton_cells_are_exact():
    """Cells with no rows contribute zero; single-level margins still work."""
    codes_a = np.array([0, 0, 0, 0])
    codes_b = np.array([2, 2, 0, 0])
    B_a = np.array([[1.5, -0.5]])
    B_b = np.array([[1.0], [2.0], [4.0]])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.5, 1.0, 1.0])

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 1, 3, score, weights)
    assert S_cell.shape == (1, 3)
    assert W_cell[0, 1] == 0.0

    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-14)
    np.testing.assert_allclose(V, X.T @ (X * weights[:, None]), rtol=1e-14)


def test_mismatched_code_shapes_raise():
    import pytest

    with pytest.raises(ValueError, match="row dimension"):
        pair_cell_moments(
            np.zeros(3, dtype=int), np.zeros(4, dtype=int), 1, 1, np.zeros(3), np.zeros(3)
        )


def test_working_score_reduces_to_raw_residual_for_canonical_link():
    rng = np.random.default_rng(2)
    n = 200
    mu = rng.uniform(0.1, 3.0, n)
    eta = np.log(mu)
    y = rng.poisson(mu).astype(float)
    w = rng.uniform(0.2, 1.5, n)

    score = working_score(y, mu, eta, w, Poisson(), LogLink())
    np.testing.assert_allclose(score, w * (y - mu), rtol=1e-14)


def test_working_score_carries_family_factor_for_noncanonical_link():
    rng = np.random.default_rng(3)
    n = 200
    mu = rng.uniform(0.5, 4.0, n)
    eta = np.log(mu)
    y = rng.gamma(2.0, mu / 2.0)
    w = rng.uniform(0.2, 1.5, n)

    score = working_score(y, mu, eta, w, Gamma(), LogLink())
    np.testing.assert_allclose(score, w * (y - mu) / mu, rtol=1e-12)
