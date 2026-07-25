"""Dense-oracle tests for compact sum-to-zero structured algebra."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.solvers.hessian_factor import DenseHessianFactor, HessianFactor
from superglm.solvers.structured import (
    SumToZeroBlockOperator,
    compact_operator_diagonal,
    materialize_compact_operator,
)
from superglm.solvers.sum_to_zero import (
    ProfiledSumToZeroBlockFactor,
    SumToZeroBlockFactor,
)
from superglm.types import PenaltyComponent


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


def _factor_fixture(
    *,
    n_levels: int,
    block_size: int,
    small_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    C = rng.normal(scale=0.18, size=(n_levels, block_size, small_size))
    D = np.empty((n_levels, block_size, block_size))
    D_inverse = np.empty_like(D)
    for level in range(n_levels):
        local_root = rng.normal(size=(block_size, block_size))
        D[level] = local_root.T @ local_root + 1.2 * np.eye(block_size)
        D_inverse[level] = np.linalg.inv(D[level])
    if small_size:
        schur_root = rng.normal(size=(small_size, small_size))
        schur = schur_root.T @ schur_root + np.eye(small_size)
        A = schur + np.einsum(
            "kiq,kij,kjr->qr",
            C,
            D_inverse,
            C,
            optimize=True,
        )
    else:
        A = np.empty((0, 0))
    small_indices = np.arange(small_size, dtype=np.intp)
    structured_indices = np.arange(
        small_size,
        small_size + (n_levels - 1) * block_size,
        dtype=np.intp,
    ).reshape(n_levels - 1, block_size)
    operator = SumToZeroBlockOperator(
        A=A,
        C=C,
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )
    dense = materialize_compact_operator(operator)
    factor = SumToZeroBlockFactor(
        A=A,
        C=C,
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="x:g:sz",
        level_labels=tuple(f"level_{index}" for index in range(n_levels)),
    )
    return factor, operator, dense


@pytest.mark.parametrize(
    ("n_levels", "block_size", "small_size", "seed"),
    [
        (2, 2, 0, 1301),
        (3, 3, 1, 1303),
        (5, 2, 3, 1307),
    ],
)
def test_sum_to_zero_factor_matches_dense_solve_inverse_and_logdet(
    n_levels: int,
    block_size: int,
    small_size: int,
    seed: int,
) -> None:
    factor, _operator, dense = _factor_fixture(
        n_levels=n_levels,
        block_size=block_size,
        small_size=small_size,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    rhs = rng.normal(size=dense.shape[0])
    rhs_matrix = rng.normal(size=(dense.shape[0], 3))
    inverse = np.linalg.inv(dense)
    selected = np.unique(np.array([0, dense.shape[0] // 2, dense.shape[0] - 1], dtype=np.intp))

    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(dense, rhs),
        rtol=2e-11,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        factor.solve(rhs_matrix),
        np.linalg.solve(dense, rhs_matrix),
        rtol=2e-11,
        atol=2e-12,
    )
    assert factor.logdet() == pytest.approx(np.linalg.slogdet(dense)[1], abs=2e-10)
    np.testing.assert_allclose(
        factor.selected_inverse_block(selected),
        inverse[np.ix_(selected, selected)],
        rtol=2e-10,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        factor.selected_inverse_diagonal(np.arange(dense.shape[0])),
        np.diag(inverse),
        rtol=2e-10,
        atol=2e-11,
    )
    assert isinstance(factor, HessianFactor)
    assert factor.rank == dense.shape[0]
    assert not factor.rank_truncated


def test_sum_to_zero_factor_accepts_identifiable_singular_local_blocks() -> None:
    D = np.array(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 3.0]],
            [[1.4, 0.2], [0.2, 1.1]],
        ]
    )
    C = np.empty((3, 2, 0))
    A = np.empty((0, 0))
    structured_indices = np.arange(4, dtype=np.intp).reshape(2, 2)
    operator = SumToZeroBlockOperator(
        A=A,
        C=C,
        D=D,
        small_indices=np.array([], dtype=np.intp),
        structured_indices=structured_indices,
    )
    dense = materialize_compact_operator(operator)

    factor = SumToZeroBlockFactor(
        A=A,
        C=C,
        D=D,
        small_indices=np.array([], dtype=np.intp),
        structured_indices=structured_indices,
        term_name="x:g:sz",
        level_labels=("thin", "medium", "rich"),
    )

    np.testing.assert_allclose(
        factor.solve(np.arange(1.0, 5.0)),
        np.linalg.solve(dense, np.arange(1.0, 5.0)),
        rtol=2e-11,
    )
    assert factor.logdet() == pytest.approx(np.linalg.slogdet(dense)[1], abs=2e-10)
    assert factor.deficient_levels == ("thin", "medium")


def test_sum_to_zero_factor_names_globally_unidentifiable_levels() -> None:
    D = np.tile(np.diag([1.0, 0.0]), (3, 1, 1))

    with pytest.raises(
        np.linalg.LinAlgError,
        match=r"x:g:sz.*globally unidentifiable.*'a'.*'b'.*'c'",
    ):
        SumToZeroBlockFactor(
            A=np.empty((0, 0)),
            C=np.empty((3, 2, 0)),
            D=D,
            small_indices=np.array([], dtype=np.intp),
            structured_indices=np.arange(4, dtype=np.intp).reshape(2, 2),
            term_name="x:g:sz",
            level_labels=("a", "b", "c"),
        )


def test_sum_to_zero_factor_is_invariant_to_which_level_is_reconstructed() -> None:
    D = np.array(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 3.0]],
            [[1.4, 0.2], [0.2, 1.1]],
        ]
    )
    rng = np.random.default_rng(1429)
    rhs = rng.normal(size=4)

    for permutation in (
        np.array([0, 1, 2]),
        np.array([2, 0, 1]),
        np.array([1, 2, 0]),
    ):
        permuted = D[permutation]
        operator = SumToZeroBlockOperator(
            A=np.empty((0, 0)),
            C=np.empty((3, 2, 0)),
            D=permuted,
            small_indices=np.array([], dtype=np.intp),
            structured_indices=np.arange(4, dtype=np.intp).reshape(2, 2),
        )
        dense = materialize_compact_operator(operator)
        factor = SumToZeroBlockFactor(
            A=np.empty((0, 0)),
            C=np.empty((3, 2, 0)),
            D=permuted,
            small_indices=np.array([], dtype=np.intp),
            structured_indices=np.arange(4, dtype=np.intp).reshape(2, 2),
            term_name="x:g:sz",
            level_labels=tuple(str(index) for index in permutation),
        )
        np.testing.assert_allclose(
            factor.solve(rhs),
            np.linalg.solve(dense, rhs),
            rtol=2e-11,
            atol=2e-12,
        )
        assert factor.logdet() == pytest.approx(np.linalg.slogdet(dense)[1], abs=2e-10)


def test_sum_to_zero_factor_raw_level_covariances_include_final_level() -> None:
    factor, _operator, dense = _factor_fixture(
        n_levels=4,
        block_size=2,
        small_size=2,
        seed=1481,
    )
    inverse = np.linalg.inv(dense)
    for level in range(factor.n_levels):
        mapping = np.zeros((factor.block_size, dense.shape[0]))
        if level < factor.n_levels - 1:
            mapping[:, factor.structured_indices[level]] = np.eye(factor.block_size)
        else:
            for indices in factor.structured_indices:
                mapping[:, indices] = -np.eye(factor.block_size)
        expected = mapping @ inverse @ mapping.T
        np.testing.assert_allclose(
            factor.raw_level_inverse_block(level),
            expected,
            rtol=2e-10,
            atol=2e-11,
        )


def test_sum_to_zero_factor_logdet_equals_absolute_raw_kkt_determinant() -> None:
    factor, operator, dense = _factor_fixture(
        n_levels=4,
        block_size=3,
        small_size=2,
        seed=1531,
    )
    q = len(operator.small_indices)
    raw_width = q + operator.n_levels * operator.block_size
    raw = np.zeros((raw_width, raw_width))
    raw[:q, :q] = operator.A
    constraint = np.zeros((operator.block_size, raw_width))
    for level in range(operator.n_levels):
        local = slice(
            q + level * operator.block_size,
            q + (level + 1) * operator.block_size,
        )
        raw[local, :q] = operator.C[level]
        raw[:q, local] = operator.C[level].T
        raw[local, local] = operator.D[level]
        constraint[:, local] = np.eye(operator.block_size)
    kkt = np.block(
        [
            [raw, constraint.T],
            [constraint, np.zeros((operator.block_size, operator.block_size))],
        ]
    )

    assert np.linalg.slogdet(kkt)[0] != 0
    assert factor.logdet() == pytest.approx(np.linalg.slogdet(kkt)[1], abs=2e-10)
    assert factor.logdet() == pytest.approx(np.linalg.slogdet(dense)[1], abs=2e-10)


def test_sum_to_zero_factor_complete_hessian_protocol_matches_dense_reference() -> None:
    factor, operator, dense = _factor_fixture(
        n_levels=4,
        block_size=2,
        small_size=2,
        seed=1601,
    )
    inverse = np.linalg.inv(dense)
    dense_factor = DenseHessianFactor(
        inverse=inverse,
        log_det=np.linalg.slogdet(dense)[1],
    )
    omega = np.array([[1.7, 0.2], [0.2, 0.9]])
    component = PenaltyComponent(
        name="x:g:sz:wiggle",
        group_name="x:g:sz",
        group_index=0,
        group_sl=slice(len(factor.small_indices), factor.shape[0]),
        omega_raw=omega,
        omega_ssp=omega,
        penalty_kind="sum_to_zero",
        repeat_count=factor.n_levels,
        block_width=factor.block_size,
    )

    assert factor.trace_inverse_penalty(component) == pytest.approx(
        dense_factor.trace_inverse_penalty(component)
    )
    assert factor.penalty_cross_trace(component, component, 1.3, 0.7) == pytest.approx(
        dense_factor.penalty_cross_trace(component, component, 1.3, 0.7)
    )
    assert factor.trace_inverse_operator(operator) == pytest.approx(
        dense_factor.trace_inverse_operator(operator)
    )
    np.testing.assert_allclose(
        factor.inverse_operator_diagonal(operator),
        dense_factor.inverse_operator_diagonal(operator),
        rtol=2e-10,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        factor.inverse_operator_square_diagonal(operator),
        dense_factor.inverse_operator_square_diagonal(operator),
        rtol=2e-10,
        atol=2e-11,
    )
    assert factor.operator_cross_trace(operator, operator) == pytest.approx(
        dense_factor.operator_cross_trace(operator, operator)
    )
    assert factor.penalty_operator_cross_trace(component, 1.9, operator) == pytest.approx(
        dense_factor.penalty_operator_cross_trace(component, 1.9, operator)
    )


def test_profiled_sum_to_zero_factor_matches_centered_dense_reference() -> None:
    augmented, operator, augmented_dense = _factor_fixture(
        n_levels=4,
        block_size=2,
        small_size=3,
        seed=1693,
    )
    sum_w = float(augmented_dense[0, 0])
    xtw = augmented_dense[0, 1:]
    centered = augmented_dense[1:, 1:] - np.outer(xtw, xtw) / sum_w
    profile = ProfiledSumToZeroBlockFactor(
        augmented_factor=augmented,
        sum_w=sum_w,
        xtw=xtw,
    )
    dense_factor = DenseHessianFactor(
        inverse=np.linalg.inv(centered),
        log_det=np.linalg.slogdet(centered)[1],
    )
    slope_operator = SumToZeroBlockOperator(
        A=operator.A[1:, 1:],
        C=operator.C[:, :, 1:],
        D=operator.D,
        small_indices=operator.small_indices[1:] - 1,
        structured_indices=operator.structured_indices - 1,
    )
    omega = np.array([[1.6, 0.1], [0.1, 0.7]])
    component = PenaltyComponent(
        name="x:g:sz:wiggle",
        group_name="x:g:sz",
        group_index=0,
        group_sl=slice(len(profile.small_indices), profile.shape[0]),
        omega_raw=omega,
        omega_ssp=omega,
        penalty_kind="sum_to_zero",
        repeat_count=profile.n_levels,
        block_width=profile.block_size,
    )
    rng = np.random.default_rng(1697)
    rhs = rng.normal(size=profile.shape[0])

    np.testing.assert_allclose(
        profile.solve(rhs),
        np.linalg.solve(centered, rhs),
        rtol=2e-10,
        atol=2e-11,
    )
    assert profile.logdet() == pytest.approx(np.linalg.slogdet(centered)[1], abs=2e-10)
    np.testing.assert_allclose(
        profile.selected_inverse_diagonal(np.arange(profile.shape[0])),
        np.diag(np.linalg.inv(centered)),
        rtol=2e-10,
        atol=2e-11,
    )
    assert profile.trace_inverse_penalty(component) == pytest.approx(
        dense_factor.trace_inverse_penalty(component)
    )
    assert profile.penalty_cross_trace(component, component, 1.2, 0.8) == pytest.approx(
        dense_factor.penalty_cross_trace(component, component, 1.2, 0.8)
    )
    assert profile.trace_inverse_operator(slope_operator) == pytest.approx(
        dense_factor.trace_inverse_operator(slope_operator)
    )
    np.testing.assert_allclose(
        profile.inverse_operator_diagonal(slope_operator),
        dense_factor.inverse_operator_diagonal(slope_operator),
        rtol=2e-10,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        profile.inverse_operator_square_diagonal(slope_operator),
        dense_factor.inverse_operator_square_diagonal(slope_operator),
        rtol=2e-10,
        atol=2e-11,
    )
    assert profile.operator_cross_trace(slope_operator, slope_operator) == pytest.approx(
        dense_factor.operator_cross_trace(slope_operator, slope_operator)
    )
    assert profile.penalty_operator_cross_trace(component, 1.4, slope_operator) == pytest.approx(
        dense_factor.penalty_operator_cross_trace(component, 1.4, slope_operator)
    )
