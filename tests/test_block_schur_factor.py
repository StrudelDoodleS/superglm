"""Exact block-Schur algebra for dominant factor-smooth terms."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.solvers.structured import (
    BlockSchurFactor,
    BlockStructuredSystem,
    BlockSymmetricOperator,
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    ProfiledBlockSchurFactor,
    _block_operator_bdlr,
    build_penalized_block_operator,
    materialize_compact_operator,
)
from superglm.types import PenaltyComponent


def _fixture(*, n_levels: int = 5, block_size: int = 3, small_size: int = 4):
    rng = np.random.default_rng(731)
    roots = rng.normal(size=(n_levels, block_size, block_size))
    D = np.einsum("kji,kjl->kil", roots, roots) + 1.2 * np.eye(block_size)[None, :, :]
    C = rng.normal(scale=0.25, size=(n_levels, block_size, small_size))
    D_inv_C = np.linalg.solve(D, C)
    schur_root = rng.normal(size=(small_size, small_size))
    Q = schur_root.T @ schur_root + 0.8 * np.eye(small_size)
    A = Q + np.einsum("kiq,kir->qr", C, D_inv_C)
    small_indices = np.arange(small_size, dtype=np.intp)
    structured_indices = np.arange(
        small_size,
        small_size + n_levels * block_size,
        dtype=np.intp,
    ).reshape(n_levels, block_size)
    operator = BlockSymmetricOperator(
        A=A,
        C=C,
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )
    factor = BlockSchurFactor(
        A=A,
        C=C,
        D=D,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="x:group:fs",
    )
    dense = materialize_compact_operator(operator)
    return rng, operator, factor, dense


@pytest.mark.parametrize("block", ["A", "C"])
def test_a_non_finite_block_schur_ordinary_or_cross_block_is_refused(block) -> None:
    """D was guarded; A and C were not, and nothing else refused them.

    The sibling of the ScalarSchurFactor case: an inf in either block passed
    every remaining check and only surfaced later as a nan logdet in the REML
    criterion. Both are typed LinAlgError so the observed-geometry build scores
    the point infeasible instead of stopping the fit.
    """
    _, _, factor, _ = _fixture()
    A, C, D = factor.A.copy(), factor.C.copy(), factor.D
    if block == "A":
        A[0, 0] = np.inf
    else:
        C[0, 0, 0] = np.inf

    with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
        BlockSchurFactor(
            A=A,
            C=C,
            D=D,
            small_indices=factor.small_indices,
            structured_indices=factor.structured_indices,
            term_name="x:group:fs",
        )


def _components(factor: BlockSchurFactor) -> tuple[PenaltyComponent, PenaltyComponent]:
    start = int(factor.structured_indices.min())
    stop = int(factor.structured_indices.max()) + 1
    repeated = PenaltyComponent(
        name="fs:wiggle",
        group_name="fs",
        group_index=1,
        group_sl=slice(start, stop),
        omega_raw=np.diag([1.4, 0.3, 0.0]),
        omega_ssp=np.diag([1.4, 0.3, 0.0]),
        rank=2.0 * factor.n_levels,
        penalty_kind="repeated",
        repeat_count=factor.n_levels,
        block_width=factor.block_size,
    )
    dense_small = PenaltyComponent(
        name="small",
        group_name="small",
        group_index=0,
        group_sl=slice(0, len(factor.small_indices)),
        omega_raw=np.diag(np.linspace(0.4, 1.0, len(factor.small_indices))),
        omega_ssp=np.diag(np.linspace(0.4, 1.0, len(factor.small_indices))),
        rank=float(len(factor.small_indices)),
    )
    return repeated, dense_small


def test_block_override_cross_validation_uses_participating_coordinate_scale() -> None:
    _rng, operator, _factor, _dense = _fixture()
    system = BlockStructuredSystem(
        operator=operator,
        xtw_small=np.zeros(len(operator.small_indices)),
        xtw_structured=np.zeros((operator.n_levels, operator.block_size)),
        xtwz_small=np.zeros(len(operator.small_indices)),
        xtwz_structured=np.zeros((operator.n_levels, operator.block_size)),
        sum_w=1.0,
        sum_wz=0.0,
        dominant_group_index=1,
        dominant_group_name="fs",
    )
    penalty = np.zeros(operator.shape)
    flat_structured = operator.structured_indices.ravel()
    penalty[operator.small_indices, operator.small_indices] = 1.0
    penalty[operator.small_indices[0], operator.small_indices[0]] = 1.0e12
    penalty[flat_structured, flat_structured] = 1.0
    penalty[flat_structured[0], operator.small_indices[1]] = 1.0e-3
    penalty[operator.small_indices[1], flat_structured[0]] = 1.0e-3

    with pytest.raises(ValueError, match="couples the dominant and dense-small blocks"):
        build_penalized_block_operator(
            system,
            [],
            [],
            0.0,
            S_override=penalty,
        )


def _expanded(component: PenaltyComponent, width: int) -> np.ndarray:
    result = np.zeros((width, width))
    if component.penalty_kind == "repeated":
        block = np.kron(np.eye(component.repeat_count), component.omega_ssp)
        result[component.group_sl, component.group_sl] = block
    elif component.penalty_kind == "identity":
        indices = np.arange(width)[component.group_sl]
        result[indices, indices] = 1.0
    else:
        result[component.group_sl, component.group_sl] = component.omega_ssp
    return result


def test_block_operator_bdlr_drops_structural_zero_low_rank_parts() -> None:
    _rng, operator, _factor, _dense = _fixture()
    pure_blocks = BlockSymmetricOperator(
        A=np.zeros_like(operator.A),
        C=np.zeros_like(operator.C),
        D=operator.D,
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )
    small_only = BlockSymmetricOperator(
        A=operator.A,
        C=np.zeros_like(operator.C),
        D=np.zeros_like(operator.D),
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )

    block_repr = _block_operator_bdlr(pure_blocks)
    small_repr = _block_operator_bdlr(small_only)

    assert block_repr.basis.shape[1] == 0
    assert block_repr.core.shape == (0, 0)
    assert small_repr.basis.shape[1] == len(operator.small_indices)
    assert small_repr.core.shape == operator.A.shape


def test_block_schur_solve_logdet_rank_and_selected_inverse_match_dense() -> None:
    rng, _operator, factor, dense = _fixture()
    rhs = rng.normal(size=dense.shape[0])
    multi_rhs = rng.normal(size=(dense.shape[0], 4))
    inverse = np.linalg.inv(dense)

    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(dense, rhs), atol=2e-12)
    np.testing.assert_allclose(
        factor.solve(multi_rhs),
        np.linalg.solve(dense, multi_rhs),
        atol=2e-12,
    )
    assert factor.logdet() == pytest.approx(np.linalg.slogdet(dense)[1], abs=2e-12)
    assert factor.rank == dense.shape[0]
    selected = np.array([0, 2, 4, 7, 13], dtype=np.intp)
    np.testing.assert_allclose(
        factor.selected_inverse_block(selected),
        inverse[np.ix_(selected, selected)],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        factor.selected_inverse_diagonal(selected),
        np.diag(inverse)[selected],
        atol=2e-12,
    )


def test_block_schur_repeated_and_dense_penalty_traces_match_dense() -> None:
    _rng, _operator, factor, dense = _fixture()
    inverse = np.linalg.inv(dense)
    repeated, dense_small = _components(factor)

    for component in (repeated, dense_small):
        omega = _expanded(component, dense.shape[0])
        assert factor.trace_inverse_penalty(component) == pytest.approx(
            np.trace(inverse @ omega),
            abs=2e-11,
        )
    for left in (repeated, dense_small):
        for right in (repeated, dense_small):
            expected = (
                1.2
                * 0.7
                * np.trace(
                    inverse
                    @ _expanded(left, dense.shape[0])
                    @ inverse
                    @ _expanded(right, dense.shape[0])
                )
            )
            assert factor.penalty_cross_trace(left, right, 1.2, 0.7) == pytest.approx(
                expected,
                abs=3e-11,
            )


def test_block_diagonal_low_rank_operator_protocol_matches_dense() -> None:
    rng, operator, factor, dense = _fixture()
    inverse = np.linalg.inv(dense)
    p = dense.shape[0]
    low_basis = rng.normal(size=(p, 2))
    low_core_raw = rng.normal(size=(2, 2))
    low_rank = LowRankSymmetricOperator(
        basis=low_basis,
        core=0.5 * (low_core_raw + low_core_raw.T),
    )
    cross = rng.normal(size=p)
    center = rng.normal(size=p)
    centered = CenteredBlockOperator(
        raw=operator,
        cross=cross,
        total=1.7,
        center=center,
    )
    repeated, _dense_small = _components(factor)

    for compact in (operator, low_rank, centered):
        materialized = materialize_compact_operator(compact)
        product = inverse @ materialized
        assert factor.trace_inverse_operator(compact) == pytest.approx(
            np.trace(product),
            abs=3e-11,
        )
        np.testing.assert_allclose(
            factor.inverse_operator_diagonal(compact),
            np.diag(product),
            atol=3e-11,
        )
        np.testing.assert_allclose(
            factor.inverse_operator_square_diagonal(compact),
            np.diag(product @ product),
            atol=5e-10,
        )

    expected_cross = np.trace(
        inverse
        @ materialize_compact_operator(centered)
        @ inverse
        @ materialize_compact_operator(low_rank)
    )
    assert factor.operator_cross_trace(centered, low_rank) == pytest.approx(
        expected_cross,
        abs=5e-10,
    )
    expected_penalty_cross = np.trace(
        inverse @ (1.3 * _expanded(repeated, p)) @ inverse @ materialize_compact_operator(centered)
    )
    assert factor.penalty_operator_cross_trace(repeated, 1.3, centered) == pytest.approx(
        expected_penalty_cross,
        abs=5e-10,
    )


def test_profiled_block_schur_adapter_matches_augmented_dense_inverse() -> None:
    rng, operator, _factor, _dense = _fixture(n_levels=4, block_size=3, small_size=3)
    K, k, q = operator.C.shape
    C_augmented = rng.normal(scale=0.2, size=(K, k, q + 1))
    D_inv_C = np.linalg.solve(operator.D, C_augmented)
    root = rng.normal(size=(q + 1, q + 1))
    Q = root.T @ root + np.diag([4.0, 0.8, 0.8, 0.8])
    A_augmented = Q + np.einsum("kiq,kir->qr", C_augmented, D_inv_C)
    small_indices = np.arange(q + 1, dtype=np.intp)
    structured_indices = np.arange(q + 1, q + 1 + K * k, dtype=np.intp).reshape(K, k)
    augmented = BlockSchurFactor(
        A=A_augmented,
        C=C_augmented,
        D=operator.D,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="x:group:fs",
    )
    slope_xtw = np.empty(q + K * k)
    slope_xtw[:q] = A_augmented[0, 1:]
    slope_xtw[q:] = C_augmented[:, :, 0].ravel()
    profiled = ProfiledBlockSchurFactor(
        augmented_factor=augmented,
        sum_w=A_augmented[0, 0],
        xtw=slope_xtw,
    )
    augmented_dense = augmented.solve(np.eye(augmented.shape[0]))
    expected = augmented_dense[1:, 1:]
    rhs = rng.normal(size=profiled.shape[0])

    np.testing.assert_allclose(profiled.solve(rhs), expected @ rhs, atol=3e-12)
    selected = np.array([0, 3, 8], dtype=np.intp)
    np.testing.assert_allclose(
        profiled.selected_inverse_block(selected),
        expected[np.ix_(selected, selected)],
        atol=3e-12,
    )
    np.testing.assert_allclose(
        profiled.selected_inverse_diagonal(selected),
        np.diag(expected)[selected],
        atol=3e-12,
    )
    assert profiled.logdet() == pytest.approx(
        augmented.logdet() - np.log(A_augmented[0, 0]),
        abs=2e-12,
    )
    repeated, dense_small = _components(profiled)
    for component in (repeated, dense_small):
        assert profiled.trace_inverse_penalty(component) == pytest.approx(
            np.trace(expected @ _expanded(component, profiled.shape[0])),
            abs=3e-11,
        )
    materialized_operator = materialize_compact_operator(operator)
    product = expected @ materialized_operator
    assert profiled.trace_inverse_operator(operator) == pytest.approx(
        np.trace(product),
        abs=3e-11,
    )
    np.testing.assert_allclose(
        profiled.inverse_operator_diagonal(operator),
        np.diag(product),
        atol=3e-11,
    )
    np.testing.assert_allclose(
        profiled.inverse_operator_square_diagonal(operator),
        np.diag(product @ product),
        atol=5e-10,
    )
    expected_cross = np.trace(
        expected @ _expanded(repeated, profiled.shape[0]) @ expected @ materialized_operator
    )
    assert profiled.penalty_operator_cross_trace(repeated, 1.0, operator) == pytest.approx(
        expected_cross,
        abs=5e-10,
    )


def test_block_schur_reports_singular_local_level() -> None:
    _rng, operator, _factor, _dense = _fixture()
    invalid = np.array(operator.D, copy=True)
    invalid[2, -1, -1] = -10.0

    with pytest.raises(np.linalg.LinAlgError, match=r"x:group:fs.*level 2.*positive definite"):
        BlockSchurFactor(
            A=operator.A,
            C=operator.C,
            D=invalid,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            term_name="x:group:fs",
        )


def test_block_schur_rejects_coupled_singular_schur_complement() -> None:
    _rng, operator, _factor, _dense = _fixture()
    D_inv_C = np.linalg.solve(operator.D, operator.C)
    singular_A = np.einsum("kiq,kir->qr", operator.C, D_inv_C)

    with pytest.raises(
        np.linalg.LinAlgError,
        match=r"x:group:fs.*coupled rank-deficient Schur null space",
    ):
        BlockSchurFactor(
            A=singular_A,
            C=operator.C,
            D=operator.D,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
            term_name="x:group:fs",
        )


def test_block_schur_refuses_large_structured_inverse_materialization() -> None:
    _rng, operator, _factor, _dense = _fixture(n_levels=20, block_size=4, small_size=2)
    factor = BlockSchurFactor(
        A=operator.A,
        C=operator.C,
        D=operator.D,
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
        term_name="x:group:fs",
        max_structured_inverse_block=16,
    )

    with pytest.raises(ValueError, match="Refusing to materialize"):
        factor.selected_inverse_block(operator.structured_indices.ravel())
