"""Tests for the shared centered numerical-rank policy."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.solvers.centered_system import build_centered_system
from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_factor, decompose_gram


def _dense_design_matrix(X: np.ndarray) -> DesignMatrix:
    return DesignMatrix([DenseGroupMatrix(X)], n=X.shape[0], p=X.shape[1])


def test_shared_rank_policy_matches_normal_equation_boundary() -> None:
    eps = np.finfo(float).eps

    assert SHARED_RANK_POLICY.factor_rcond == pytest.approx(np.sqrt(eps))
    assert SHARED_RANK_POLICY.gram_rcond == eps
    assert SHARED_RANK_POLICY.certification_band == 32.0
    assert SHARED_RANK_POLICY.warning_condition == pytest.approx(1.0 / np.sqrt(eps))
    assert SHARED_RANK_POLICY.severe_condition == pytest.approx(1.0 / eps)


def test_centered_system_avoids_raw_moment_cancellation() -> None:
    X = np.column_stack((np.full(8, 7.0), 1e9 + np.arange(8, dtype=float)))
    W = np.ones(8)
    z = 2.0 + np.arange(8, dtype=float)

    system = build_centered_system(
        dm=_dense_design_matrix(X),
        W=W,
        z_off=z,
        penalty=np.zeros((2, 2)),
    )

    centered = X - np.average(X, axis=0, weights=W)
    np.testing.assert_allclose(system.data_gram, centered.T @ (W[:, None] * centered))
    assert system.data_gram[0, 0] == pytest.approx(0.0, abs=1e-13)
    assert system.data_gram[1, 1] == pytest.approx(42.0)
    np.testing.assert_allclose(system.rhs, centered.T @ (W * (z - np.average(z, weights=W))))


def test_centered_rhs_is_stable_with_large_feature_and_response_means() -> None:
    delta = np.arange(12, dtype=float) - 5.5
    X = np.column_stack((1e12 + delta, -3e11 + 2.0 * delta))
    z = 8e12 - 4.0 * delta
    W = np.linspace(0.5, 2.0, len(delta))

    system = build_centered_system(
        dm=_dense_design_matrix(X),
        W=W,
        z_off=z,
        penalty=np.eye(2),
    )

    Xc = X - np.average(X, axis=0, weights=W)
    zc = z - np.average(z, weights=W)
    np.testing.assert_allclose(system.data_gram, Xc.T @ (W[:, None] * Xc))
    np.testing.assert_allclose(system.rhs, Xc.T @ (W * zc))
    np.testing.assert_allclose(system.hessian, system.data_gram + np.eye(2))
    for values in (
        system.mean_x,
        system.data_gram,
        system.rhs,
        system.penalty,
        system.hessian,
    ):
        assert not values.flags.writeable


def test_centered_system_requires_positive_total_weight() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_centered_system(
            dm=_dense_design_matrix(np.ones((3, 1))),
            W=np.zeros(3),
            z_off=np.ones(3),
            penalty=np.zeros((1, 1)),
        )


def test_centered_system_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_centered_system(
            dm=_dense_design_matrix(np.ones((3, 1))),
            W=-np.ones(3),
            z_off=np.ones(3),
            penalty=np.zeros((1, 1)),
        )


def test_identity_uses_full_rank_cholesky_and_exact_operations() -> None:
    decomposition = decompose_gram(np.eye(3))

    assert decomposition.method == "cholesky"
    assert decomposition.rank == 3
    assert not decomposition.rank_truncated
    rhs = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(decomposition.solve(rhs), rhs)
    np.testing.assert_allclose(decomposition.pseudo_inverse(), np.eye(3))
    assert decomposition.log_pdet == pytest.approx(0.0)


def test_exact_duplicate_is_truncated_consistently() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    decomposition = decompose_gram(matrix)

    assert decomposition.rank == 1
    assert decomposition.rank_truncated
    inverse = decomposition.pseudo_inverse()
    np.testing.assert_allclose(matrix @ inverse @ matrix, matrix, atol=1e-12)
    assert not decomposition.is_estimable(np.array([1.0, 0.0]))
    assert decomposition.is_estimable(np.array([1.0, 1.0]))


def test_shared_boundary_retains_above_and_truncates_below() -> None:
    eps = SHARED_RANK_POLICY.gram_rcond

    below = decompose_gram(np.array([[1.0, 1.0 - eps], [1.0 - eps, 1.0]]))
    above = decompose_gram(np.array([[1.0, 1.0 - 8 * eps], [1.0 - 8 * eps, 1.0]]))

    assert below.rank == 1
    assert above.rank == 2


def test_factor_and_gram_rules_agree_at_normal_equation_boundary() -> None:
    eps = SHARED_RANK_POLICY.gram_rcond
    gram = np.array([[1.0, 1.0 - eps], [1.0 - eps, 1.0]])
    factor = np.linalg.cholesky(gram).T

    factor_decomposition = decompose_factor(factor)
    gram_decomposition = decompose_gram(gram)

    assert factor_decomposition.rank == gram_decomposition.rank == 1


def test_column_rescaling_preserves_rank_and_fitted_projection() -> None:
    base = np.array([[2.0, 0.3], [0.3, 1.0]])
    rhs = np.array([1.0, -2.0])
    base_solution = decompose_gram(base).solve(rhs)

    scale = np.diag([1e-12, 1e12])
    scaled = scale @ base @ scale
    scaled_rhs = scale @ rhs
    scaled_solution = decompose_gram(scaled).solve(scaled_rhs)

    assert decompose_gram(base).rank == decompose_gram(scaled).rank == 2
    np.testing.assert_allclose(scale @ scaled_solution, base_solution, rtol=1e-10)


def test_zero_diagonal_column_is_inactive_and_nonestimable() -> None:
    decomposition = decompose_gram(np.diag([2.0, 0.0]))

    assert decomposition.rank == 1
    np.testing.assert_allclose(decomposition.solve(np.array([4.0, 9.0])), [2.0, 0.0])
    assert not decomposition.is_estimable(np.array([0.0, 1.0]))
