"""Dense-oracle tests for compact sum-to-zero structured algebra."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.solvers.hessian_factor import DenseHessianFactor, HessianFactor
from superglm.solvers.rank import decompose_gram
from superglm.solvers.structured import (
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    _multiply_symmetric_bdlr_coalesced,
    _operator_bdlr,
    centered_operator_coefficient_estimable,
    compact_operator_diagonal,
    materialize_compact_operator,
)
from superglm.solvers.sum_to_zero import (
    ProfiledSumToZeroBlockFactor,
    SumToZeroBlockFactor,
    _decompose_local_psd_batch,
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


def test_local_psd_decomposition_batches_levels_and_preserves_rank() -> None:
    blocks = np.array(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 3.0]],
            [[1.4, 0.2], [0.2, 1.1]],
        ]
    )

    locals_, minimum = _decompose_local_psd_batch(
        blocks,
        term_name="x:g:sz",
        level_labels=("thin", "medium", "rich"),
    )

    assert tuple(local.rank for local in locals_) == (1, 1, 2)
    assert minimum == pytest.approx(0.0, abs=1e-15)
    for block, local in zip(blocks, locals_, strict=True):
        np.testing.assert_allclose(
            block @ local.pinv @ block,
            block,
            rtol=2e-12,
            atol=2e-12,
        )


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


def test_bdlr_product_coalesces_repeated_low_rank_bases() -> None:
    factor, operator, dense = _factor_fixture(
        n_levels=5,
        block_size=2,
        small_size=3,
        seed=1663,
    )
    inverse = factor._inverse_bdlr()
    compact = _operator_bdlr(operator, factor.structured_indices)

    product = _multiply_symmetric_bdlr_coalesced(inverse, compact)

    expected_rank_bound = inverse.basis.shape[1] + compact.basis.shape[1]
    assert product.left.shape[1] <= expected_rank_bound
    assert product.right.shape[1] <= expected_rank_bound
    materialized = np.zeros_like(dense)
    for level, indices in enumerate(product.structured_indices):
        materialized[np.ix_(indices, indices)] = product.blocks[level]
    materialized += product.left @ product.core @ product.right.T
    np.testing.assert_allclose(
        materialized,
        np.linalg.solve(dense, dense),
        rtol=3e-10,
        atol=3e-11,
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


def test_centered_sum_to_zero_estimability_matches_dense_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 5
    rows_per_level = 12
    block_size = 2
    codes = np.repeat(np.arange(n_levels), rows_per_level)
    x = np.tile(np.linspace(-1.0, 1.0, rows_per_level), n_levels)
    z = x**2 + 0.03 * codes
    weights = 0.7 + np.linspace(0.0, 0.6, len(codes))
    small = np.column_stack((np.ones(len(codes)), z))
    local_basis = np.column_stack((np.ones(len(codes)), x))

    level_masks = tuple(codes == level for level in range(n_levels))
    C = np.stack(
        [local_basis[mask].T @ (weights[mask, None] * small[mask]) for mask in level_masks]
    )
    D = np.stack(
        [local_basis[mask].T @ (weights[mask, None] * local_basis[mask]) for mask in level_masks]
    )
    raw_structured_cross = np.stack([local_basis[mask].T @ weights[mask] for mask in level_masks])

    structured_indices = np.arange(
        small.shape[1],
        small.shape[1] + (n_levels - 1) * block_size,
        dtype=np.intp,
    ).reshape(n_levels - 1, block_size)
    free_structured = np.zeros((len(codes), structured_indices.size))
    for level, mask in enumerate(level_masks[:-1]):
        free_structured[mask, level * block_size : (level + 1) * block_size] = local_basis[mask]
    final_mask = level_masks[-1]
    free_structured[final_mask] = np.tile(-local_basis[final_mask], n_levels - 1)
    public_design = np.column_stack((small, free_structured))
    cross = public_design.T @ weights

    raw = SumToZeroBlockOperator(
        A=small.T @ (weights[:, None] * small),
        C=C,
        D=D,
        small_indices=np.arange(small.shape[1], dtype=np.intp),
        structured_indices=structured_indices,
    )
    operator = CenteredBlockOperator(
        raw=raw,
        cross=cross,
        total=float(np.sum(weights)),
        center=cross / np.sum(weights),
        raw_structured_cross=raw_structured_cross,
    )
    dense = public_design.T @ (weights[:, None] * public_design)
    dense -= np.outer(cross, cross) / np.sum(weights)
    expected = decompose_gram(dense).coefficient_estimable()

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("full-rank SZ estimability must remain on the compact path")

    monkeypatch.setattr(
        "superglm.solvers.structured._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(centered_operator_coefficient_estimable(operator), expected)
    assert not expected[0]
