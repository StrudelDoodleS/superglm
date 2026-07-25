"""Dense-oracle tests for compact sum-to-zero structured algebra."""

from __future__ import annotations

import numpy as np

from superglm.solvers.structured import (
    SumToZeroBlockOperator,
    compact_operator_diagonal,
    materialize_compact_operator,
)


def _sum_to_zero_operator_fixture():
    rng = np.random.default_rng(1147)
    n_levels = 4
    block_size = 2
    small_size = 3
    public_width = small_size + (n_levels - 1) * block_size
    small_indices = np.array([0, 4, 8], dtype=np.intp)
    structured_indices = np.array([[1, 2], [3, 5], [6, 7]], dtype=np.intp)

    root = rng.normal(size=(small_size, small_size))
    A = root.T @ root + 2.0 * np.eye(small_size)
    C = rng.normal(scale=0.2, size=(n_levels, block_size, small_size))
    D = np.empty((n_levels, block_size, block_size))
    for level in range(n_levels):
        local_root = rng.normal(size=(block_size, block_size))
        D[level] = local_root.T @ local_root + 1.5 * np.eye(block_size)

    raw_width = small_size + n_levels * block_size
    raw_hessian = np.zeros((raw_width, raw_width))
    raw_hessian[:small_size, :small_size] = A
    for level in range(n_levels):
        raw_sl = slice(
            small_size + level * block_size,
            small_size + (level + 1) * block_size,
        )
        raw_hessian[raw_sl, :small_size] = C[level]
        raw_hessian[:small_size, raw_sl] = C[level].T
        raw_hessian[raw_sl, raw_sl] = D[level]

    transform = np.zeros((raw_width, public_width))
    transform[:small_size, small_indices] = np.eye(small_size)
    for level, indices in enumerate(structured_indices):
        raw_sl = slice(
            small_size + level * block_size,
            small_size + (level + 1) * block_size,
        )
        transform[raw_sl, indices] = np.eye(block_size)
    final_sl = slice(
        small_size + (n_levels - 1) * block_size,
        small_size + n_levels * block_size,
    )
    for indices in structured_indices:
        transform[final_sl, indices] = -np.eye(block_size)

    expected = transform.T @ raw_hessian @ transform
    operator = SumToZeroBlockOperator(
        A=A,
        C=C,
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )
    return operator, expected


def test_sum_to_zero_operator_matches_dense_free_coordinates() -> None:
    operator, expected = _sum_to_zero_operator_fixture()
    rng = np.random.default_rng(1261)
    rhs = rng.normal(size=operator.shape[0])
    rhs_matrix = rng.normal(size=(operator.shape[0], 3))

    np.testing.assert_allclose(operator.matvec(rhs), expected @ rhs, atol=1e-12)
    np.testing.assert_allclose(
        operator.matvec(rhs_matrix),
        expected @ rhs_matrix,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compact_operator_diagonal(operator),
        np.diag(expected),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        materialize_compact_operator(operator),
        expected,
        atol=1e-12,
    )
