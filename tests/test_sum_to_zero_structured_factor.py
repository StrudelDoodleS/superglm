"""Dense-oracle tests for compact sum-to-zero structured algebra."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.solvers.structured import (
    SumToZeroBlockOperator,
    SumToZeroLeafSystem,
    _orthonormal_column_span,
    build_penalized_sum_to_zero_operator,
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


def test_sum_to_zero_override_cross_validation_uses_participating_coordinate_scale() -> None:
    operator, _expected = _sum_to_zero_operator_fixture()
    free_levels = operator.n_levels - 1
    block_size = operator.block_size
    system = SumToZeroLeafSystem(
        operator=operator,
        leaf=None,
        xtw_small=np.zeros(len(operator.small_indices)),
        xtw_structured=np.zeros((free_levels, block_size)),
        xtwz_small=np.zeros(len(operator.small_indices)),
        xtwz_structured=np.zeros((free_levels, block_size)),
        raw_xtw_structured=np.zeros((operator.n_levels, block_size)),
        sum_w=1.0,
        sum_wz=0.0,
        dominant_group_index=1,
        dominant_group_name="sz",
    )
    penalty = np.zeros(operator.shape)
    penalty[operator.small_indices, operator.small_indices] = 1.0
    penalty[operator.small_indices[0], operator.small_indices[0]] = 1.0e12
    local = np.eye(block_size)
    structured_penalty = np.empty((free_levels * block_size,) * 2)
    for left in range(free_levels):
        left_slice = slice(left * block_size, (left + 1) * block_size)
        for right in range(free_levels):
            right_slice = slice(right * block_size, (right + 1) * block_size)
            structured_penalty[left_slice, right_slice] = (2.0 if left == right else 1.0) * local
    flat_structured = operator.structured_indices.ravel()
    penalty[np.ix_(flat_structured, flat_structured)] = structured_penalty
    penalty[flat_structured[0], operator.small_indices[1]] = 1.0e-3
    penalty[operator.small_indices[1], flat_structured[0]] = 1.0e-3

    with pytest.raises(ValueError, match="couples the SZ and dense-small blocks"):
        build_penalized_sum_to_zero_operator(
            system,
            [],
            [],
            0.0,
            S_override=penalty,
        )


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


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_sz_operator_block_is_refused_as_curvature(bad) -> None:
    """The operator refuses one step before the factor does, and needs the same class.

    ``SumToZeroBlockOperator`` is built inside ``build_structured_system``, which
    the observed-geometry build calls before it assembles any factor. That
    refusal is iterate-conditioned for the same reason the factor's is -- these
    blocks are this iterate's weighted moments -- so it carries the same class,
    and the build wraps that call in the seam that scores the point infeasible.
    A plain ValueError there would escape the fit instead.
    """
    n_levels, block_size = 3, 2
    D = np.tile(np.diag([2.0, 3.0]), (n_levels, 1, 1))
    D[1, 0, 0] = bad

    with pytest.raises(np.linalg.LinAlgError, match="must be finite"):
        SumToZeroBlockOperator(
            A=np.empty((0, 0)),
            C=np.empty((n_levels, block_size, 0)),
            D=D,
            small_indices=np.empty(0, dtype=np.intp),
            structured_indices=np.arange((n_levels - 1) * block_size, dtype=np.intp).reshape(
                n_levels - 1, block_size
            ),
        )


def test_orthonormal_column_span_is_invariant_to_candidate_scale() -> None:
    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1e-14],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )

    span = _orthonormal_column_span(candidates)

    assert span.shape == (4, 2)
    np.testing.assert_allclose(span @ span.T, np.diag([1.0, 1.0, 0.0, 0.0]))


# ``tilt**2`` sets the lifted null eigenvalue and these two points bracket
# ``eigh``'s bar at this width (32, so the bar is ``32 eps`` against a bare
# ``gram_rcond`` of ``eps``).  Measured on this fixture over 7
# ``OPENBLAS_CORETYPE`` microkernels at one thread, as a multiple of the
# largest eigenvalue: the exact-null design reads -3.032e-16 to +1.494e-16 --
# BOTH SIGNS, magnitude 0.32x to 1.37x of the bare cut, which is issue #356's
# coin flip in this module; ``1.3e-6`` reads 2.017e-15 to 2.199e-15, i.e. 9.1x
# to 9.9x the bare cut and 0.28x to 0.31x of the floored one; ``1e-5`` reads
# 1.232e-13 to 1.235e-13, 17.3x ABOVE the floored cut.
_SZ_SUB_BAR_TILT = 1.3e-6
_SZ_RESOLVED_TILT = 1e-5
