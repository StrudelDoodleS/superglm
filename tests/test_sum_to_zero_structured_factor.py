"""Dense-oracle tests for compact sum-to-zero structured algebra."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg

from superglm.solvers.hessian_factor import DenseHessianFactor, HessianFactor
from superglm.solvers.rank import (
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
)
from superglm.solvers.structured import (
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    SumToZeroBlockStructuredSystem,
    _certified_ritz_discarded,
    _multiply_symmetric_bdlr_coalesced,
    _operator_bdlr,
    _orthonormal_column_span,
    _sum_to_zero_inherent_null_row_norms,
    _sum_to_zero_public_null_geometry,
    _sum_to_zero_public_spectral_bound,
    _sum_to_zero_scaled_basis_null_row_norms,
    build_penalized_sum_to_zero_operator,
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


def _sum_to_zero_free_design(local_factors: np.ndarray) -> np.ndarray:
    n_levels, rows_per_level, block_size = local_factors.shape
    free_structured = np.zeros((n_levels * rows_per_level, (n_levels - 1) * block_size))
    for level, factor in enumerate(local_factors[:-1]):
        row_slice = slice(level * rows_per_level, (level + 1) * rows_per_level)
        column_slice = slice(level * block_size, (level + 1) * block_size)
        free_structured[row_slice, column_slice] = factor
    final_rows = slice((n_levels - 1) * rows_per_level, n_levels * rows_per_level)
    free_structured[final_rows] = np.hstack([-local_factors[-1]] * (n_levels - 1))
    return free_structured


def _sum_to_zero_centered_operator(
    local_factors: np.ndarray,
    small: np.ndarray,
) -> tuple[CenteredBlockOperator, np.ndarray]:
    n_levels, rows_per_level, block_size = local_factors.shape
    free_structured = _sum_to_zero_free_design(local_factors)
    public_design = np.column_stack((small, free_structured))
    cross = public_design.T @ np.ones(len(public_design))
    C = np.stack(
        [
            factor.T @ small[level * rows_per_level : (level + 1) * rows_per_level]
            for level, factor in enumerate(local_factors)
        ]
    )
    D = np.stack([factor.T @ factor for factor in local_factors])
    raw_structured_cross = np.stack(
        [factor.T @ np.ones(rows_per_level) for factor in local_factors]
    )
    A = small.T @ small
    raw = SumToZeroBlockOperator(
        A=0.5 * (A + A.T),
        C=C,
        D=D,
        small_indices=np.arange(small.shape[1], dtype=np.intp),
        structured_indices=np.arange(
            small.shape[1],
            small.shape[1] + (n_levels - 1) * block_size,
            dtype=np.intp,
        ).reshape(n_levels - 1, block_size),
    )
    return (
        CenteredBlockOperator(
            raw=raw,
            cross=cross,
            total=float(len(public_design)),
            center=cross / len(public_design),
            raw_structured_cross=raw_structured_cross,
        ),
        public_design,
    )


def _sum_to_zero_public_factor(
    local_factors: tuple[np.ndarray, ...],
    small_factor: np.ndarray,
) -> np.ndarray:
    """Map heterogeneous raw block factors into public sum-to-zero coordinates."""
    small_factor = np.asarray(small_factor, dtype=np.float64)
    n_levels = len(local_factors)
    block_size = local_factors[0].shape[1]
    small_width = small_factor.shape[1]
    raw_width = small_width + n_levels * block_size
    public_width = small_width + (n_levels - 1) * block_size
    row_count = small_factor.shape[0] + sum(factor.shape[0] for factor in local_factors)
    raw_factor = np.zeros((row_count, raw_width))
    cursor = small_factor.shape[0]
    raw_factor[:cursor, :small_width] = small_factor
    for level, factor in enumerate(local_factors):
        next_cursor = cursor + factor.shape[0]
        columns = slice(
            small_width + level * block_size,
            small_width + (level + 1) * block_size,
        )
        raw_factor[cursor:next_cursor, columns] = factor
        cursor = next_cursor

    transform = np.zeros((raw_width, public_width))
    transform[:small_width, :small_width] = np.eye(small_width)
    for level in range(n_levels - 1):
        raw_columns = slice(
            small_width + level * block_size,
            small_width + (level + 1) * block_size,
        )
        public_columns = slice(
            small_width + level * block_size,
            small_width + (level + 1) * block_size,
        )
        transform[raw_columns, public_columns] = np.eye(block_size)
    final_columns = slice(
        small_width + (n_levels - 1) * block_size,
        small_width + n_levels * block_size,
    )
    for level in range(n_levels - 1):
        public_columns = slice(
            small_width + level * block_size,
            small_width + (level + 1) * block_size,
        )
        transform[final_columns, public_columns] = -np.eye(block_size)
    return raw_factor @ transform


def _sum_to_zero_diagonal_moment_operator(D: np.ndarray) -> CenteredBlockOperator:
    n_levels, block_size, _ = D.shape
    structured_indices = np.arange(
        (n_levels - 1) * block_size,
        dtype=np.intp,
    ).reshape(n_levels - 1, block_size)
    raw = SumToZeroBlockOperator(
        A=np.empty((0, 0)),
        C=np.empty((n_levels, block_size, 0)),
        D=D,
        small_indices=np.empty(0, dtype=np.intp),
        structured_indices=structured_indices,
    )
    cross = np.zeros(raw.shape[0])
    return CenteredBlockOperator(
        raw=raw,
        cross=cross,
        total=1.0,
        center=cross,
        raw_structured_cross=np.zeros((n_levels, block_size)),
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
    system = SumToZeroBlockStructuredSystem(
        operator=operator,
        xtw_small=np.zeros(len(operator.small_indices)),
        xtw_structured=np.zeros((free_levels, block_size)),
        xtwz_small=np.zeros(len(operator.small_indices)),
        xtwz_structured=np.zeros((free_levels, block_size)),
        raw_xtw_structured=np.zeros((operator.n_levels, block_size)),
        raw_xtwz_structured=np.zeros((operator.n_levels, block_size)),
        sum_w=1.0,
        sum_wz=0.0,
        dominant_group_index=1,
        dominant_group_name="sz",
        level_labels=tuple(range(operator.n_levels)),
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


@pytest.mark.parametrize(
    ("n_levels", "block_size", "deficient_levels", "seed", "scaled"),
    [
        pytest.param(300, 10, 0, 20260727, False, id="full-rank-300x10"),
        pytest.param(300, 10, 40, 20260728, False, id="mixed-rank-300x10"),
        pytest.param(129, 4, 20, 20260729, True, id="scaled-mixed-rank-129x4"),
    ],
)
def test_local_psd_batch_matches_rank_reconstruction_and_null_geometry(
    n_levels: int,
    block_size: int,
    deficient_levels: int,
    seed: int,
    scaled: bool,
) -> None:
    rng = np.random.default_rng(seed)
    eigenvectors, _triangular = np.linalg.qr(rng.normal(size=(n_levels, block_size, block_size)))
    planted_eigenvalues = rng.uniform(0.5, 2.0, size=(n_levels, block_size))
    if deficient_levels:
        planted_eigenvalues[:deficient_levels, -1] = 0.0
    if scaled:
        scales = 10.0 ** rng.uniform(-3.0, 3.0, size=n_levels)
        planted_eigenvalues *= scales[:, None]
    blocks = np.einsum(
        "kij,kj,klj->kil",
        eigenvectors,
        planted_eigenvalues,
        eigenvectors,
        optimize=True,
    )
    labels = tuple(f"level-{level}" for level in range(n_levels))

    locals_, minimum = _decompose_local_psd_batch(
        blocks,
        term_name="x:g:sz",
        level_labels=labels,
    )

    reference_eigenvalues, _reference_eigenvectors = np.linalg.eigh(blocks)
    scales = np.maximum(np.max(np.abs(reference_eigenvalues), axis=1), 1.0)
    thresholds = np.finfo(np.float64).eps * block_size * scales * 10.0
    expected_positive = reference_eigenvalues > thresholds[:, None]
    assert minimum == pytest.approx(float(np.min(reference_eigenvalues)), abs=1.0e-12)
    for level, (block, local) in enumerate(zip(blocks, locals_, strict=True)):
        expected_values = reference_eigenvalues[level][expected_positive[level]]
        assert local.rank == int(np.count_nonzero(expected_positive[level]))
        np.testing.assert_allclose(
            block @ local.pinv @ block,
            block,
            rtol=3.0e-9,
            atol=max(2.0e-9, thresholds[level] * 50.0),
        )
        np.testing.assert_allclose(
            block @ local.null,
            0.0,
            rtol=0.0,
            atol=max(2.0e-9, thresholds[level] * 50.0),
        )
        assert np.sum(np.log(local.positive_eigenvalues)) == pytest.approx(
            np.sum(np.log(expected_values)),
            rel=2.0e-11,
            abs=2.0e-10,
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
    centered_design = public_design - np.average(public_design, axis=0, weights=weights)
    expected = decompose_factor(np.sqrt(weights)[:, None] * centered_design).coefficient_estimable()

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("full-rank SZ estimability must remain on the compact path")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(centered_operator_coefficient_estimable(operator), expected)
    assert not expected[0]


@pytest.mark.parametrize("n_deficient_levels", [1, 258])
def test_wide_deficient_sum_to_zero_estimability_matches_dense_compactly(
    monkeypatch: pytest.MonkeyPatch,
    n_deficient_levels: int,
) -> None:
    n_levels = 258
    block_size = 2
    small_indices = np.array([0], dtype=np.intp)
    structured_indices = np.arange(
        1,
        1 + (n_levels - 1) * block_size,
        dtype=np.intp,
    ).reshape(n_levels - 1, block_size)
    D = np.tile(np.eye(block_size), (n_levels, 1, 1))
    D[:n_deficient_levels] = np.diag([1.0, 0.0])
    raw = SumToZeroBlockOperator(
        A=np.array([[2.0]]),
        C=np.zeros((n_levels, block_size, 1)),
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )
    cross = np.zeros(raw.shape[0])
    operator = CenteredBlockOperator(
        raw=raw,
        cross=cross,
        total=1.0,
        center=cross,
        raw_structured_cross=np.zeros((n_levels, block_size)),
    )
    expected = decompose_gram(materialize_compact_operator(operator)).coefficient_estimable()

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("wide deficient SZ estimability must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    actual = centered_operator_coefficient_estimable(operator)

    np.testing.assert_array_equal(actual, expected)
    assert actual[small_indices[0]]


def test_wide_deficient_sum_to_zero_uses_public_factor_local_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 130
    block_size = 4
    small_indices = np.array([0], dtype=np.intp)
    structured_indices = np.arange(
        1,
        1 + (n_levels - 1) * block_size,
        dtype=np.intp,
    ).reshape(n_levels - 1, block_size)
    local_factor = np.random.default_rng(3).normal(size=(2, block_size))
    identity_factors = tuple(np.eye(block_size) for _level in range(1, n_levels))
    public_factor = _sum_to_zero_public_factor(
        (local_factor, *identity_factors),
        np.array([[np.sqrt(2.0)]]),
    )
    expected = decompose_factor(public_factor).coefficient_estimable()
    assert decompose_factor(local_factor).rank == 2

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("wide deficient SZ estimability must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    D = np.tile(np.eye(block_size), (n_levels, 1, 1))
    D[0] = local_factor.T @ local_factor
    raw = SumToZeroBlockOperator(
        A=np.array([[2.0]]),
        C=np.zeros((n_levels, block_size, 1)),
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )
    cross = np.zeros(raw.shape[0])
    operator = CenteredBlockOperator(
        raw=raw,
        cross=cross,
        total=1.0,
        center=cross,
        raw_structured_cross=np.zeros((n_levels, block_size)),
    )
    assert needs_factor_certification(decompose_gram(D[0]))
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected,
    )


def test_wide_deficient_sum_to_zero_schur_rank_uses_augmented_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 258
    block_size = 2
    codes = np.repeat(np.arange(n_levels), block_size)
    local_basis = np.tile(np.eye(block_size), (n_levels, 1))
    local_basis[: 2 * block_size] = np.tile(
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        (2, 1),
    )
    level_masks = tuple(codes == level for level in range(n_levels))
    free_structured = np.zeros((len(codes), (n_levels - 1) * block_size))
    for level, mask in enumerate(level_masks[:-1]):
        block_slice = slice(level * block_size, (level + 1) * block_size)
        free_structured[mask, block_slice] = local_basis[mask]
    free_structured[level_masks[-1]] = np.tile(
        -local_basis[level_masks[-1]],
        n_levels - 1,
    )

    alias_map = np.zeros((free_structured.shape[1], 2))
    alias_map[:8] = np.array(
        [
            [0.0, 1.0],
            [-1.0, -3.0],
            [1.0, -3.0],
            [-2.0, 2.0],
            [0.0, 1.0],
            [3.0, -2.0],
            [2.0, 2.0],
            [-1.0, -2.0],
        ]
    )
    small = free_structured @ alias_map
    public_design = np.column_stack((small, free_structured))
    cross = public_design.T @ np.ones(len(codes))
    C = np.stack(
        [local_basis[mask].T @ small[mask] for mask in level_masks],
    )
    D = np.stack(
        [local_basis[mask].T @ local_basis[mask] for mask in level_masks],
    )
    raw_structured_cross = np.stack(
        [local_basis[mask].T @ np.ones(np.count_nonzero(mask)) for mask in level_masks],
    )
    raw = SumToZeroBlockOperator(
        A=small.T @ small,
        C=C,
        D=D,
        small_indices=np.arange(2, dtype=np.intp),
        structured_indices=np.arange(
            2,
            2 + (n_levels - 1) * block_size,
            dtype=np.intp,
        ).reshape(n_levels - 1, block_size),
    )
    operator = CenteredBlockOperator(
        raw=raw,
        cross=cross,
        total=float(len(codes)),
        center=cross / len(codes),
        raw_structured_cross=raw_structured_cross,
    )
    centered_design = public_design - np.mean(public_design, axis=0)
    expected = decompose_factor(centered_design).coefficient_estimable()
    assert not np.any(expected[raw.small_indices])

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("wide deficient SZ estimability must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    actual = centered_operator_coefficient_estimable(operator)

    np.testing.assert_array_equal(actual, expected)


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


def test_ritz_rank_certificate_rejects_cutoff_crossing_residual() -> None:
    operator = scipy.sparse.linalg.LinearOperator(
        (2, 2),
        matvec=lambda values: np.array([0.0, values[1]]),
        dtype=np.float64,
    )
    eigenvalues = np.array([0.0])
    exact_vector = np.array([[1.0], [0.0]])
    np.testing.assert_array_equal(
        _certified_ritz_discarded(
            operator,
            eigenvalues,
            exact_vector,
            np.finfo(np.float64).eps,
        ),
        np.array([True]),
    )

    unresolved_vector = np.array([[np.sqrt(1.0 - 1e-16)], [1e-8]])
    with pytest.raises(np.linalg.LinAlgError, match="residual crosses"):
        _certified_ritz_discarded(
            operator,
            eigenvalues,
            unresolved_vector,
            np.finfo(np.float64).eps,
        )


def test_sum_to_zero_public_spectral_bound_covers_the_structured_gram() -> None:
    rng = np.random.default_rng(8712)
    local_factors = rng.normal(size=(7, 5, 3))
    small = rng.normal(size=(35, 2))
    operator, _public_design = _sum_to_zero_centered_operator(local_factors, small)
    raw = operator.raw
    assert isinstance(raw, SumToZeroBlockOperator)
    structured_scale = np.sqrt(compact_operator_diagonal(operator)[raw.structured_indices])

    bound = _sum_to_zero_public_spectral_bound(operator, structured_scale)
    structured_indices = raw.structured_indices.ravel()
    structured_gram = materialize_compact_operator(operator)[
        np.ix_(structured_indices, structured_indices)
    ]
    flat_scale = structured_scale.ravel()
    normalized_gram = structured_gram / np.outer(flat_scale, flat_scale)

    assert np.linalg.eigvalsh(normalized_gram)[-1] <= bound * (
        1.0 + 10.0 * np.finfo(np.float64).eps
    )


def test_wide_sum_to_zero_certifies_local_factor_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 129
    block_size = 4
    local_factor = np.random.default_rng(1).normal(size=(3, block_size))
    local_factors = np.tile(local_factor, (n_levels, 1, 1))
    small = np.tile(local_factor[:, :2], (n_levels, 1))
    operator, public_design = _sum_to_zero_centered_operator(local_factors, small)
    expected_estimable = decompose_factor(
        public_design - np.mean(public_design, axis=0)
    ).coefficient_estimable()
    assert decompose_factor(local_factor).rank == 3
    assert np.all(expected_estimable[: small.shape[1]])

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("wide certification-limited SZ inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    assert needs_factor_certification(decompose_gram(operator.raw.D[0]))
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected_estimable,
    )


def test_sum_to_zero_certifies_full_local_blocks_in_public_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_basis = np.eye(6, 5)
    local_basis -= np.mean(local_basis, axis=0)
    local_factors = np.stack(
        [local_basis.copy() for _level in range(5)]
        + [local_basis * np.array([1e8, 1.0, 1.0, 1.0, 1.0])]
    )
    operator, public_design = _sum_to_zero_centered_operator(
        local_factors,
        np.empty((36, 0)),
    )
    expected = decompose_factor(public_design)
    local_decompositions = tuple(decompose_gram(factor.T @ factor) for factor in local_factors)

    assert all(decomposition.rank == 5 for decomposition in local_decompositions)
    assert not any(
        needs_factor_certification(decomposition) for decomposition in local_decompositions
    )
    assert expected.rank == 21
    np.testing.assert_array_equal(
        np.flatnonzero(~expected.coefficient_estimable()),
        np.arange(0, 25, 5),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("public-coordinate SZ certification must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


@pytest.mark.parametrize("n_levels", [5, 40])
def test_sum_to_zero_weak_candidates_are_ranked_at_gram_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    n_levels: int,
) -> None:
    D = np.ones((n_levels, 1, 1))
    D[-1, 0, 0] = 1e9
    operator = _sum_to_zero_diagonal_moment_operator(D)
    expected = decompose_gram(materialize_compact_operator(operator))
    assert np.all(expected.coefficient_estimable())

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("weak SZ candidates must be certified compactly")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


@pytest.mark.parametrize("n_levels", [5, 258])
def test_sum_to_zero_positive_columns_are_certified_with_structural_zeros(
    monkeypatch: pytest.MonkeyPatch,
    n_levels: int,
) -> None:
    D = np.zeros((n_levels, 2, 2))
    D[:-1, 1, 1] = 1.0
    D[-1, 1, 1] = 1e16
    operator = _sum_to_zero_diagonal_moment_operator(D)
    expected = np.zeros(operator.shape[0], dtype=bool)
    if n_levels == 5:
        np.testing.assert_array_equal(
            decompose_gram(materialize_compact_operator(operator)).coefficient_estimable(),
            expected,
        )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("positive SZ columns must remain compact with structural zeros")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected,
    )


def test_sum_to_zero_exact_nulls_do_not_saturate_weak_rank_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 40
    D = np.zeros((n_levels, 2, 2))
    D[:-1, 1, 1] = 1.0
    D[-1] = np.diag([1.0, 1e9])
    operator = _sum_to_zero_diagonal_moment_operator(D)
    expected = np.tile(np.array([False, True]), n_levels - 1)

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("exact SZ nulls must be deflated before weak rank certification")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected,
    )


def test_sum_to_zero_active_null_geometry_survives_structural_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 40
    D = np.zeros((n_levels, 4, 4))
    D[:-1] = np.diag([0.0, 0.0, 1.0, 1.0])
    D[-1] = np.diag([0.0, 1.0, 1e9, 1.0])
    operator = _sum_to_zero_diagonal_moment_operator(D)
    expected = np.tile(np.array([False, False, True, True]), n_levels - 1)

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("active SZ null geometry must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected,
    )


def test_sum_to_zero_exact_projection_noise_is_not_an_additional_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 40
    D = np.ones((n_levels, 2, 2)) * np.eye(2)
    D[:2, 0, 0] = 0.0
    D[-1, 1, 1] = 1e9
    operator = _sum_to_zero_diagonal_moment_operator(D)
    expected = np.ones((n_levels - 1, 2), dtype=bool)
    expected[:2, 0] = False
    expected = expected.ravel()

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("exact-null projection noise must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected,
    )


def test_sum_to_zero_structural_null_norm_uses_factor_cutoff() -> None:
    local_null_projector = np.zeros((3, 2, 2))
    geometry = _sum_to_zero_public_null_geometry(
        local_null_projector,
        np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    row_norm, cutoff, _ambiguous = _sum_to_zero_inherent_null_row_norms(geometry)

    np.testing.assert_array_equal(row_norm, np.zeros((2, 2)))
    assert cutoff == np.sqrt(np.finfo(np.float64).eps)


def test_wide_sum_to_zero_deflated_inverse_handles_heterogeneous_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(18)
    n_levels = 13
    block_size = 4
    rows_per_level = 6
    local_factors = []
    for _level in range(n_levels):
        left = rng.normal(size=(rows_per_level, 3))
        right = rng.normal(size=(3, block_size))
        coordinate_scale = 10.0 ** rng.uniform(-6.0, 6.0, size=block_size)
        local_factors.append((left @ right) * coordinate_scale)
    small = rng.normal(size=(n_levels * rows_per_level, 3))
    operator, public_design = _sum_to_zero_centered_operator(
        np.stack(local_factors),
        small,
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    np.testing.assert_array_equal(
        np.flatnonzero(expected.coefficient_estimable()),
        np.array([0, 1, 2, 22]),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("deflated SZ shift-invert must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_small_sum_to_zero_spectrum_filters_gram_eigenspace_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(4)
    n_levels = 16
    block_size = 2
    rows_per_level = 4
    local_factors = []
    for _level in range(n_levels):
        local_rank = int(rng.integers(1, 3))
        left = rng.normal(size=(rows_per_level, local_rank))
        right = rng.normal(size=(local_rank, block_size))
        coordinate_scale = 10.0 ** rng.uniform(-4.0, 4.0, size=block_size)
        local_factors.append((left @ right) * coordinate_scale)
    small = rng.normal(size=(n_levels * rows_per_level, 3))
    operator, public_design = _sum_to_zero_centered_operator(
        np.stack(local_factors),
        small,
    )
    centered_design = public_design - np.mean(public_design, axis=0)
    expected = decompose_factor(centered_design)
    gram_expected = decompose_gram(centered_design.T @ centered_design)
    np.testing.assert_array_equal(
        gram_expected.coefficient_estimable(),
        expected.coefficient_estimable(),
    )

    fallback_calls = 0

    def certified_dense_fallback(
        fallback_operator: CenteredBlockOperator,
    ) -> np.ndarray:
        nonlocal fallback_calls
        fallback_calls += 1
        return decompose_gram(
            materialize_compact_operator(fallback_operator)
        ).coefficient_estimable()

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        certified_dense_fallback,
    )
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )
    assert fallback_calls == 1


def test_wide_sum_to_zero_public_rank_certificate_stays_block_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 258
    block_size = 3
    structured_indices = np.arange(
        (n_levels - 1) * block_size,
        dtype=np.intp,
    ).reshape(n_levels - 1, block_size)
    D = np.tile(np.eye(block_size), (n_levels, 1, 1))
    D[-1, 0, 0] = 1e16
    raw = SumToZeroBlockOperator(
        A=np.empty((0, 0)),
        C=np.empty((n_levels, block_size, 0)),
        D=D,
        small_indices=np.empty(0, dtype=np.intp),
        structured_indices=structured_indices,
    )
    cross = np.zeros(raw.shape[0])
    operator = CenteredBlockOperator(
        raw=raw,
        cross=cross,
        total=1.0,
        center=cross,
        raw_structured_cross=np.zeros((n_levels, block_size)),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("wide public-coordinate SZ certification must remain block-bounded")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    actual = centered_operator_coefficient_estimable(operator).reshape(
        n_levels - 1,
        block_size,
    )
    assert not np.any(actual[:, 0])
    assert np.all(actual[:, 1:])


def test_sum_to_zero_constraint_rank_uses_scaled_parameter_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centered_row = np.array([-1.0, 0.0, 1.0])
    local_factors = np.stack(
        (
            np.outer(centered_row, np.array([1e8, 1e8])),
            np.outer(centered_row, np.array([1.0, 0.0])),
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                ]
            ),
        )
    )
    operator, public_design = _sum_to_zero_centered_operator(
        local_factors,
        np.empty((9, 0)),
    )
    expected = decompose_factor(public_design)

    assert [decompose_factor(factor).rank for factor in local_factors] == [1, 1, 2]
    np.testing.assert_array_equal(
        expected.coefficient_estimable(),
        np.array([False, False, True, True]),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("parameter-scaled SZ constraint rank must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_wide_sum_to_zero_null_span_is_invariant_to_coordinate_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 129
    block_size = 4
    rng = np.random.default_rng(4)
    local_factor = rng.normal(size=(4, 2)) @ rng.normal(size=(2, block_size))
    local_factor *= np.array([1.0, 1e-5, 1e-10, 1e-14])
    local_factors = np.tile(local_factor, (n_levels, 1, 1))
    small = np.tile(local_factor[:, :2], (n_levels, 1))
    operator, public_design = _sum_to_zero_centered_operator(local_factors, small)
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))

    assert decompose_factor(local_factor).rank == 2
    assert not np.any(expected.coefficient_estimable()[small.shape[1] :])

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("scaled wide SZ inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_full_rank_sum_to_zero_schur_exact_alias_uses_factor_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 5
    block_size = 2
    rng = np.random.default_rng(2)
    local_factors = rng.normal(size=(n_levels, block_size, block_size))
    free_structured = _sum_to_zero_free_design(local_factors)
    alias_map = rng.normal(size=(free_structured.shape[1], 3))
    operator, public_design = _sum_to_zero_centered_operator(
        local_factors,
        free_structured @ alias_map,
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    assert not np.any(expected.coefficient_estimable()[: alias_map.shape[1]])

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("full-rank SZ inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_sum_to_zero_scale_separated_aliases_preserve_null_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 4
    block_size = 2
    rng = np.random.default_rng(0)
    local_factors = rng.normal(size=(n_levels, block_size, block_size))
    local_factors *= np.array([1.0, 1e6])[None, None, :]
    free_structured = _sum_to_zero_free_design(local_factors)
    alias_map = rng.normal(size=(free_structured.shape[1], 3))
    alias_map *= np.array([1.0, 1e-5, 1e-10])
    operator, public_design = _sum_to_zero_centered_operator(
        local_factors,
        free_structured @ alias_map,
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    assert not np.any(expected.coefficient_estimable())

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("scale-separated SZ inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_sum_to_zero_heterogeneous_null_projector_uses_solve_error_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(42)
    angle_1, angle_2 = rng.uniform(-np.pi, np.pi, size=2)
    delta = 1e-1
    direction_1 = np.array([np.cos(angle_1), np.sin(angle_1), 0.0, 0.0, 0.0])
    direction_2 = np.array([-np.sin(angle_1), np.cos(angle_1), 0.0, 0.0, 0.0])
    direction_3 = np.array([0.0, 0.0, np.cos(angle_2), np.sin(angle_2), 0.0])
    direction_4 = np.array([0.0, 0.0, -np.sin(angle_2), np.cos(angle_2), 0.0])
    direction_5 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    local_ranges = (
        np.stack((direction_3, direction_4, direction_5)),
        np.stack((direction_2, direction_4, direction_5)),
        np.stack((direction_1, direction_3, direction_5)),
        np.stack(
            (
                direction_1,
                direction_2,
                (direction_5 - delta * direction_4) / np.sqrt(1.0 + delta**2),
            )
        ),
    )
    coordinate_scale = 10.0 ** rng.uniform(-1.0, 1.0, size=5)
    local_factors = []
    for local_range in local_ranges:
        row_map = rng.normal(size=(6, 3))
        row_map -= np.mean(row_map, axis=0)
        local_factors.append((row_map @ local_range) * coordinate_scale)
    local_factors_array = np.stack(local_factors)
    operator, public_design = _sum_to_zero_centered_operator(
        local_factors_array,
        np.empty((24, 0)),
    )
    expected = decompose_factor(public_design)
    np.testing.assert_allclose(np.mean(public_design, axis=0), 0.0, atol=1e-15)
    np.testing.assert_array_equal(
        np.flatnonzero(expected.coefficient_estimable()),
        np.array([2, 3, 4, 9, 12, 13, 14]),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("heterogeneous-null SZ inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_sum_to_zero_scaled_null_leverage_preserves_resolved_small_residual() -> None:
    delta = 1e-7
    retained_row_space = np.array([[np.sqrt(1.0 - delta**2), delta]])
    scaled_bases = (
        np.array([[1.0], [0.0]]),
        np.array([[1.0], [0.0]]),
    )

    row_norms, null_dimension, ambiguous = _sum_to_zero_scaled_basis_null_row_norms(
        scaled_bases,
        retained_row_space,
    )

    assert null_dimension == 1
    assert ambiguous[0, 0]
    assert row_norms[0, 0] > np.sqrt(np.finfo(np.float64).eps)
    np.testing.assert_allclose(row_norms[1, 0], np.sqrt(1.0 - delta**2))


@pytest.mark.parametrize(
    ("seed", "scale_bound", "expected_estimable"),
    [
        (131011, 4.0, np.array([], dtype=np.intp)),
        (121016, 3.0, np.array([0, 4], dtype=np.intp)),
    ],
)
def test_sum_to_zero_scaled_null_leverage_matches_public_factor(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    scale_bound: float,
    expected_estimable: np.ndarray,
) -> None:
    rng = np.random.default_rng(seed)
    local_factors = []
    for rank in (2, 3, 4):
        left = rng.normal(size=(7, rank))
        right = rng.normal(size=(rank, 5))
        coordinate_scale = 10.0 ** rng.uniform(-scale_bound, scale_bound, size=5)
        local_factors.append((left @ right) * coordinate_scale)
    operator, public_design = _sum_to_zero_centered_operator(
        np.stack(local_factors),
        np.empty((21, 0)),
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    np.testing.assert_array_equal(
        np.flatnonzero(expected.coefficient_estimable()),
        expected_estimable,
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("scaled SZ null inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_sum_to_zero_lifted_null_uses_design_column_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_factor = np.array(
        [
            [1e-2, 0.0],
            [0.0, 1.0],
            [-1e-2, -1.0],
        ]
    )
    local_factors = np.tile(local_factor, (2, 1, 1))
    free_structured = _sum_to_zero_free_design(local_factors)
    small = (1e-7 * free_structured[:, 0] + free_structured[:, 1])[:, None]
    operator, public_design = _sum_to_zero_centered_operator(local_factors, small)
    expected = decompose_factor(public_design)
    np.testing.assert_array_equal(
        expected.coefficient_estimable(),
        np.array([False, True, False]),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("equilibrated SZ lifted-null inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_sum_to_zero_inherent_null_uses_public_design_column_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(8279)
    local_factors = []
    for _level in range(4):
        row = rng.normal(size=3)
        row -= np.mean(row)
        coefficients = rng.normal(size=2) * 10.0 ** rng.uniform(-3.0, 3.0, size=2)
        local_factors.append(np.outer(row, coefficients))
    local_factors_array = np.stack(local_factors)
    operator, public_design = _sum_to_zero_centered_operator(
        local_factors_array,
        np.empty((12, 0)),
    )
    expected = decompose_factor(public_design)
    np.testing.assert_allclose(np.mean(public_design, axis=0), 0.0, atol=1e-12)
    np.testing.assert_array_equal(
        expected.coefficient_estimable(),
        np.array([False, True, False, False, False, False]),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("equilibrated inherent SZ null inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )
