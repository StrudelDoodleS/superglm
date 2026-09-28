"""Exact scalar Schur-factor algebra tests."""

import importlib.util
from collections import Counter
from dataclasses import FrozenInstanceError
from importlib import import_module

import numpy as np
import pytest

from superglm.distributions import Poisson
from superglm.reml.gradient import reml_direct_hessian
from superglm.solvers._structured import factors as factors_module
from superglm.solvers.hessian_factor import DenseHessianFactor, HessianFactor, _component_omega
from superglm.solvers.rank import decompose_factor, decompose_gram, needs_factor_certification
from superglm.solvers.structured import (
    BlockSchurFactor,
    BlockSymmetricOperator,
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
    SumBlockOperator,
    SymmetricBlockOperator,
    centered_operator_coefficient_estimable,
    materialize_compact_operator,
)
from superglm.types import PenaltyComponent


def _spd_scalar_blocks():
    rng = np.random.default_rng(519)
    small_indices = np.array([0, 2, 6], dtype=np.intp)
    structured_indices = np.array([1, 3, 4, 5], dtype=np.intp)
    C = rng.normal(scale=0.3, size=(4, 3))
    d = rng.uniform(1.2, 2.0, size=4)
    root = rng.normal(size=(3, 3))
    Q = root.T @ root + np.eye(3)
    A = Q + C.T @ (C / d[:, None])
    H = np.zeros((7, 7))
    H[np.ix_(small_indices, small_indices)] = A
    H[np.ix_(structured_indices, small_indices)] = C
    H[np.ix_(small_indices, structured_indices)] = C.T
    H[structured_indices, structured_indices] = d
    return A, C, d, small_indices, structured_indices, H


def _independent_centered_operator(
    local_factors: np.ndarray,
    small: np.ndarray,
) -> tuple[CenteredBlockOperator, np.ndarray]:
    n_levels, rows_per_level, block_size = local_factors.shape
    structured = np.zeros((n_levels * rows_per_level, n_levels * block_size))
    for level, factor in enumerate(local_factors):
        row_slice = slice(level * rows_per_level, (level + 1) * rows_per_level)
        column_slice = slice(level * block_size, (level + 1) * block_size)
        structured[row_slice, column_slice] = factor
    public_design = np.column_stack((small, structured))
    cross = public_design.T @ np.ones(len(public_design))
    C = np.stack(
        [
            factor.T @ small[level * rows_per_level : (level + 1) * rows_per_level]
            for level, factor in enumerate(local_factors)
        ]
    )
    D = np.stack([factor.T @ factor for factor in local_factors])
    A = small.T @ small
    raw = BlockSymmetricOperator(
        A=0.5 * (A + A.T),
        C=C,
        D=D,
        small_indices=np.arange(small.shape[1], dtype=np.intp),
        structured_indices=np.arange(
            small.shape[1],
            small.shape[1] + n_levels * block_size,
            dtype=np.intp,
        ).reshape(n_levels, block_size),
    )
    return (
        CenteredBlockOperator(
            raw=raw,
            cross=cross,
            total=float(len(public_design)),
            center=cross / len(public_design),
        ),
        public_design,
    )


def _unit_level_centered_operator(width: int) -> CenteredBlockOperator:
    local_factors = np.ones((width, 1, 1), dtype=np.float64)
    small = np.empty((width, 0), dtype=np.float64)
    operator, _public_design = _independent_centered_operator(
        local_factors,
        small,
    )
    return operator


@pytest.mark.parametrize("part", ["basis", "core"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_low_rank_part_is_refused_as_curvature(part, bad) -> None:
    """Both parts and both values, because the two values failed differently.

    ``reml_w_correction`` builds the basis as ``column_stack(dmean_i, dmean_j)``
    with no check of any kind, and the core as ``[[0, -sum_w], [-sum_w, 0]]``.
    Before the guard a NaN core failed ``allclose`` against its own transpose
    and was refused as *asymmetric* -- the right refusal under the wrong name --
    while an inf core passed it (matching infs compare equal) and was accepted
    outright, carrying a non-finite low-rank update into ``_operator_dlr``. The
    NaN arm alone would pin only the ordering, leaving the silent-acceptance
    case free to come back.
    """
    rng = np.random.default_rng(4)
    basis = rng.normal(size=(7, 2))
    core = np.array([[0.0, -1.5], [-1.5, 0.0]])
    if part == "basis":
        basis[0, 0] = bad
    else:
        core = np.array([[0.0, bad], [bad, 0.0]])

    with pytest.raises(np.linalg.LinAlgError, match="must be finite"):
        LowRankSymmetricOperator(basis=basis, core=core)


@pytest.mark.parametrize("block", ["A", "C"])
def test_a_non_finite_scalar_schur_block_is_refused_rather_than_absorbed(block) -> None:
    """An inf in A or C used to build a factor carrying a nan scale.

    Only ``d`` was guarded. ``np.linalg.norm(..., ord=2)`` returns nan for an
    inf rather than raising, so nothing downstream refused either -- the factor
    was constructed and the nan travelled into the REML criterion as a logdet.
    That is worse than a refusal of the wrong class: a wrong class still stops
    something, while a nan logdet quietly distorts smoothing-parameter
    selection with every accuracy metric left looking healthy.
    """
    A, C, d, _, _, _ = _spd_scalar_blocks()
    if block == "A":
        A = A.copy()
        A[0, 0] = np.inf
    else:
        C = C.copy()
        C[0, 0] = np.inf

    with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
        ScalarSchurFactor(
            A=A,
            C=C,
            d=d,
            small_indices=np.arange(3, dtype=np.intp),
            structured_indices=np.arange(3, 7, dtype=np.intp),
            term_name="broker",
        )


def _contiguous_scalar_factor():
    A, C, d, _, _, _ = _spd_scalar_blocks()
    small_indices = np.arange(3, dtype=np.intp)
    structured_indices = np.arange(3, 7, dtype=np.intp)
    H = np.zeros((7, 7))
    H[np.ix_(small_indices, small_indices)] = A
    H[np.ix_(structured_indices, small_indices)] = C
    H[np.ix_(small_indices, structured_indices)] = C.T
    H[structured_indices, structured_indices] = d
    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="broker",
    )
    return factor, H


def test_structured_solver_module_exists():
    assert importlib.util.find_spec("superglm.solvers.structured") is not None


def test_hessian_factor_module_exists():
    assert importlib.util.find_spec("superglm.solvers.hessian_factor") is not None


def test_dense_hessian_factor_and_protocol_are_available():
    factors = import_module("superglm.solvers.hessian_factor")

    assert hasattr(factors, "HessianFactor")
    assert hasattr(factors, "DenseHessianFactor")


def test_scalar_schur_factor_is_available():
    structured = import_module("superglm.solvers.structured")

    assert hasattr(structured, "ScalarSchurFactor")


def test_symmetric_block_operator_is_available():
    structured = import_module("superglm.solvers.structured")

    assert hasattr(structured, "SymmetricBlockOperator")


def test_scalar_schur_solve_and_logdet_match_dense_factorization():
    A, C, d, small_indices, structured_indices, H = _spd_scalar_blocks()
    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="broker",
    )
    rhs = np.arange(1.0, 8.0)
    rhs_matrix = np.column_stack([rhs, rhs[::-1]])

    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(H, rhs))
    np.testing.assert_allclose(factor.solve(rhs_matrix), np.linalg.solve(H, rhs_matrix))
    np.testing.assert_allclose(factor.logdet(), np.linalg.slogdet(H)[1])
    assert factor.shape == H.shape
    assert factor.backend == "structured"
    assert isinstance(factor, HessianFactor)
    assert factor.dominant_group_name == "broker"
    assert factor.minimum_local_diagonal == pytest.approx(np.min(d))
    assert np.isfinite(factor.schur_condition_estimate)
    assert factor.fallback_reason is None
    assert not factor.used_dense_fallback


def test_scalar_schur_supports_no_dense_small_block():
    d = np.array([1.2, 2.3, 4.1])
    indices = np.arange(3, dtype=np.intp)
    factor = ScalarSchurFactor(
        A=np.empty((0, 0)),
        C=np.empty((3, 0)),
        d=d,
        small_indices=np.array([], dtype=np.intp),
        structured_indices=indices,
        term_name="policy",
    )
    rhs = np.array([0.5, -2.0, 3.0])
    identity = PenaltyComponent(
        name="policy",
        group_name="policy",
        group_index=0,
        group_sl=slice(0, 3),
        omega_raw=None,
        penalty_kind="identity",
    )

    np.testing.assert_allclose(factor.solve(rhs), rhs / d)
    np.testing.assert_allclose(factor.logdet(), np.sum(np.log(d)))
    np.testing.assert_allclose(factor.selected_inverse_diagonal(indices), 1.0 / d)
    np.testing.assert_allclose(factor.trace_inverse_penalty(identity), np.sum(1.0 / d))
    assert not factor.used_dense_fallback


def test_scalar_schur_supports_one_dense_small_column():
    A = np.array([[2.5]])
    C = np.array([[0.2], [-0.1], [0.3]])
    d = np.array([1.4, 1.7, 2.1])
    small_indices = np.array([2], dtype=np.intp)
    structured_indices = np.array([0, 1, 3], dtype=np.intp)
    H = np.zeros((4, 4))
    H[np.ix_(small_indices, small_indices)] = A
    H[np.ix_(structured_indices, small_indices)] = C
    H[np.ix_(small_indices, structured_indices)] = C.T
    H[structured_indices, structured_indices] = d
    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="territory",
    )

    rhs = np.arange(1.0, 5.0)
    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(H, rhs))
    np.testing.assert_allclose(factor.logdet(), np.linalg.slogdet(H)[1])


def test_scalar_schur_diagnostics_name_invalid_local_diagonal_and_value():
    with pytest.raises(
        np.linalg.LinAlgError,
        match=r"broker.*minimum local diagonal.*-0.25",
    ):
        ScalarSchurFactor(
            A=np.array([[1.0]]),
            C=np.array([[0.1], [0.2]]),
            d=np.array([1.0, -0.25]),
            small_indices=np.array([0], dtype=np.intp),
            structured_indices=np.array([1, 2], dtype=np.intp),
            term_name="broker",
        )


def test_scalar_schur_uses_diagnostic_small_svd_fallback_for_singular_schur():
    factor = ScalarSchurFactor(
        A=np.diag([2.0, 0.0]),
        C=np.zeros((3, 2)),
        d=np.array([1.0, 1.5, 2.0]),
        small_indices=np.array([0, 1], dtype=np.intp),
        structured_indices=np.array([2, 3, 4], dtype=np.intp),
        term_name="broker",
    )
    H = np.diag([2.0, 0.0, 1.0, 1.5, 2.0])
    rhs = np.arange(1.0, 6.0)

    np.testing.assert_allclose(factor.solve(rhs), np.linalg.pinv(H) @ rhs)
    np.testing.assert_allclose(factor.logdet(), np.log(2.0) + np.log(1.5) + np.log(2.0))
    assert factor.used_dense_fallback
    assert "Cholesky" in factor.fallback_reason
    assert np.isinf(factor.schur_condition_estimate)
    np.testing.assert_array_equal(
        factor.coefficient_estimable(),
        [True, False, True, True, True],
    )


def _block_schur(A):
    return BlockSchurFactor(
        A=A,
        C=np.zeros((2, 2, 2)),
        D=np.broadcast_to(np.eye(2), (2, 2, 2)).copy(),
        small_indices=np.array([0, 1], dtype=np.intp),
        structured_indices=np.arange(2, 6, dtype=np.intp).reshape(2, 2),
        term_name="x:group:fs",
    )


def test_schur_rank_floor_is_computed_only_on_the_svd_fallback(monkeypatch):
    """The floor's two spectral norms are full SVDs of the border.

    Built eagerly they dominated wide-border fits (a 4,000-column border spent
    12 of 14 profile samples in them) while a Schur complement Cholesky accepts
    never reads the floor.
    """
    factors = import_module("superglm.solvers._structured.factors")
    real_cutoff = factors._schur_fallback_cutoff
    calls = []

    def counted(*args):
        calls.append(args)
        return real_cutoff(*args)

    monkeypatch.setattr(factors, "_schur_fallback_cutoff", counted)
    A, C, d, small_indices, structured_indices, H = _spd_scalar_blocks()
    accepted = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="broker",
    )
    accepted_block = _block_schur(np.eye(2))
    assert not accepted.used_dense_fallback and not accepted_block.used_dense_fallback
    assert calls == []

    singular = ScalarSchurFactor(
        A=np.diag([2.0, 0.0]),
        C=np.zeros((3, 2)),
        d=np.array([1.0, 1.5, 2.0]),
        small_indices=np.array([0, 1], dtype=np.intp),
        structured_indices=np.array([2, 3, 4], dtype=np.intp),
        term_name="broker",
    )
    singular_block = _block_schur(np.diag([1.0, 0.0]))
    assert singular.used_dense_fallback and singular_block.used_dense_fallback
    assert len(calls) == 2
    assert singular.rank == 4 and singular_block.rank == 5


def test_scalar_schur_keeps_exact_tiny_decoupled_pivot():
    """A pivot that is merely tiny is not cancellation residue.

    One ordinary-block column carries a near-zero decoupled diagonal entry --
    the working-Gram signature of an unpenalized level whose weights have
    collapsed under separation, nested inside one heavy local level.  Its
    Schur pivot equals its own diagonal exactly (nothing large is subtracted
    at that coordinate), so the factorization must keep the Cholesky path,
    full rank, and the exact positive-definite log-determinant even though
    the pivot sits far below any floor scaled by the global norms.
    """
    rng = np.random.default_rng(11)
    q, k = 6, 12
    M = rng.standard_normal((q + 3, q))
    A = (M.T @ M + np.diag(np.full(q, 50.0))) * 1e4
    tiny = 1e-9
    A[-1, :] = 0.0
    A[:, -1] = 0.0
    A[-1, -1] = tiny
    d = rng.uniform(1e3, 1e5, k)
    C = rng.standard_normal((k, q)) * np.sqrt(d)[:, None] * 1e-2
    C[:, -1] = 0.0
    C[0, -1] = tiny

    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=np.arange(q, dtype=np.intp),
        structured_indices=np.arange(q, q + k, dtype=np.intp),
        term_name="collapsed",
    )
    H = np.zeros((q + k, q + k))
    H[:q, :q] = A
    H[q:, q:] = np.diag(d)
    H[q:, :q] = C
    H[:q, q:] = C.T
    sign, expected_logdet = np.linalg.slogdet(H)

    assert not factor.used_dense_fallback
    assert factor.rank == q + k
    assert not factor.rank_truncated
    assert sign > 0
    assert factor.logdet() == pytest.approx(expected_logdet, rel=1e-9)
    assert bool(np.all(factor.coefficient_estimable()))


def test_scalar_schur_rejects_cancellation_created_coupled_null_space():
    rng = np.random.default_rng(0)
    d = 10.0 ** rng.uniform(-6.0, 6.0, 42)
    C = d[:, None]
    A = np.array([[np.sum(d)]])

    with pytest.raises(
        np.linalg.LinAlgError,
        match="coupled rank-deficient Schur null space",
    ):
        ScalarSchurFactor(
            A=A,
            C=C,
            d=d,
            small_indices=np.array([0], dtype=np.intp),
            structured_indices=np.arange(1, len(d) + 1, dtype=np.intp),
            term_name="group",
        )


def test_block_schur_rejects_cancellation_created_coupled_null_space():
    rng = np.random.default_rng(0)
    d = 10.0 ** rng.uniform(-6.0, 6.0, 42)
    C = d[:, None, None]
    D = d[:, None, None]
    A = np.array([[np.sum(d)]])

    with pytest.raises(
        np.linalg.LinAlgError,
        match="coupled rank-deficient Schur null space",
    ):
        BlockSchurFactor(
            A=A,
            C=C,
            D=D,
            small_indices=np.array([0], dtype=np.intp),
            structured_indices=np.arange(1, len(d) + 1, dtype=np.intp)[:, None],
            term_name="factor_smooth",
        )


def test_block_schur_keeps_exact_tiny_decoupled_pivot():
    """A block-path pivot that is merely tiny is not cancellation residue.

    Block twin of the scalar keep test above: one ordinary-block column
    carries a near-zero decoupled diagonal entry, so its Schur pivot equals
    its own diagonal exactly and nothing large is subtracted at that
    coordinate.  The factorization must keep the Cholesky path, full rank,
    and the exact positive-definite log-determinant even though the pivot
    sits far below any floor scaled by the global norms; a global floor
    reroutes this geometry to the truncating SVD fallback, which drops the
    direction and publishes the wrong log-determinant.
    """
    rng = np.random.default_rng(11)
    q, k, b = 6, 8, 3
    M = rng.standard_normal((q + 3, q))
    A = (M.T @ M + np.diag(np.full(q, 50.0))) * 1e4
    tiny = 1e-9
    A[-1, :] = 0.0
    A[:, -1] = 0.0
    A[-1, -1] = tiny
    X = rng.standard_normal((k, b, b + 2))
    D = np.einsum("kij,klj->kil", X, X) * 1e3 + np.eye(b)[None, :, :] * 1e3
    C = rng.standard_normal((k, b, q)) * 10.0
    C[:, :, -1] = 0.0
    C[0, 0, -1] = tiny

    factor = BlockSchurFactor(
        A=A,
        C=C,
        D=D,
        small_indices=np.arange(q, dtype=np.intp),
        structured_indices=np.arange(q, q + k * b, dtype=np.intp).reshape(k, b),
        term_name="collapsed_block",
    )
    n = q + k * b
    H = np.zeros((n, n))
    H[:q, :q] = A
    for level in range(k):
        start = q + level * b
        H[start : start + b, start : start + b] = D[level]
        H[start : start + b, :q] = C[level]
        H[:q, start : start + b] = C[level].T
    sign, expected_logdet = np.linalg.slogdet(H)

    assert not factor.used_dense_fallback
    assert factor.rank == n
    assert not factor.rank_truncated
    assert sign > 0
    assert factor.logdet() == pytest.approx(expected_logdet, rel=1e-9)
    assert bool(np.all(factor.coefficient_estimable()))


def test_symmetric_block_operator_is_frozen_and_owns_read_only_arrays():
    A, C, d, small_indices, structured_indices, _ = _spd_scalar_blocks()
    operator = SymmetricBlockOperator(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )

    with pytest.raises(FrozenInstanceError):
        operator.A = np.eye(3)
    with pytest.raises(ValueError, match="read-only"):
        operator.d[0] = 0.0


def test_dense_hessian_factor_wraps_existing_inverse_contract():
    _, _, _, _, _, H = _spd_scalar_blocks()
    inverse = np.linalg.inv(H)
    logdet = np.linalg.slogdet(H)[1]
    factor = DenseHessianFactor(inverse=inverse, log_det=logdet)
    rhs = np.arange(1.0, 8.0)
    selected = np.array([5, 0, 3], dtype=np.intp)

    assert isinstance(factor, HessianFactor)
    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(H, rhs))
    np.testing.assert_allclose(
        factor.selected_inverse_block(selected),
        inverse[np.ix_(selected, selected)],
    )
    np.testing.assert_allclose(
        factor.selected_inverse_diagonal(selected),
        np.diag(inverse)[selected],
    )
    np.testing.assert_allclose(factor.logdet(), logdet)
    assert factor.backend == "dense"


def test_scalar_schur_selected_inverse_blocks_and_diagonal_match_dense_inverse():
    A, C, d, small_indices, structured_indices, H = _spd_scalar_blocks()
    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="broker",
    )
    inverse = np.linalg.inv(H)

    for selected in (
        np.array([0, 2, 6]),
        np.array([1, 4]),
        np.array([0, 1, 4, 6]),
    ):
        np.testing.assert_allclose(
            factor.selected_inverse_block(selected),
            inverse[np.ix_(selected, selected)],
        )

    selected_diagonal = np.array([5, 0, 3, 2], dtype=np.intp)
    np.testing.assert_allclose(
        factor.selected_inverse_diagonal(selected_diagonal),
        np.diag(inverse)[selected_diagonal],
    )


def test_scalar_schur_refuses_large_structured_inverse_block():
    A, C, d, small_indices, structured_indices, _ = _spd_scalar_blocks()
    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="broker",
        max_structured_inverse_block=2,
    )

    with pytest.raises(ValueError, match="request its diagonal"):
        factor.selected_inverse_block(structured_indices[:3])


def test_scalar_schur_trace_inverse_operator_matches_dense_arbitrary_sign_matrix():
    A, C, d, small_indices, structured_indices, H = _spd_scalar_blocks()
    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
        term_name="broker",
    )
    operator_A = np.array(
        [
            [0.5, -0.2, 0.1],
            [-0.2, -0.3, 0.4],
            [0.1, 0.4, 0.2],
        ]
    )
    operator_C = np.array(
        [
            [0.2, -0.1, 0.0],
            [-0.3, 0.2, 0.1],
            [0.1, 0.0, -0.2],
            [0.4, -0.1, 0.3],
        ]
    )
    operator_d = np.array([0.3, -0.2, 0.5, -0.4])
    operator = SymmetricBlockOperator(
        A=operator_A,
        C=operator_C,
        d=operator_d,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )
    dense_operator = np.zeros_like(H)
    dense_operator[np.ix_(small_indices, small_indices)] = operator_A
    dense_operator[np.ix_(structured_indices, small_indices)] = operator_C
    dense_operator[np.ix_(small_indices, structured_indices)] = operator_C.T
    dense_operator[structured_indices, structured_indices] = operator_d

    expected = np.trace(np.linalg.inv(H) @ dense_operator)
    dense_factor = DenseHessianFactor(
        inverse=np.linalg.inv(H),
        log_det=np.linalg.slogdet(H)[1],
    )

    np.testing.assert_allclose(factor.trace_inverse_operator(operator), expected)
    np.testing.assert_allclose(dense_factor.trace_inverse_operator(operator), expected)


def test_dense_and_structured_penalty_traces_match_materialized_formulas():
    structured_factor, H = _contiguous_scalar_factor()
    inverse = np.linalg.inv(H)
    dense_factor = DenseHessianFactor(inverse=inverse, log_det=np.linalg.slogdet(H)[1])
    identity = PenaltyComponent(
        name="broker",
        group_name="broker",
        group_index=1,
        group_sl=slice(3, 7),
        omega_raw=None,
        penalty_kind="identity",
    )
    omega = np.array([[1.5, 0.2], [0.2, 0.8]])
    dense_penalty = PenaltyComponent(
        name="spline",
        group_name="spline",
        group_index=0,
        group_sl=slice(0, 2),
        omega_raw=omega,
        omega_ssp=omega,
    )

    expected_identity_trace = np.trace(inverse[3:7, 3:7])
    expected_dense_trace = np.trace(inverse[0:2, 0:2] @ omega)
    identity_matrix = np.zeros_like(H)
    identity_matrix[3:7, 3:7] = np.eye(4)
    dense_matrix = np.zeros_like(H)
    dense_matrix[0:2, 0:2] = omega
    expected_identity_self = np.trace(inverse @ identity_matrix @ inverse @ identity_matrix)
    expected_cross = np.trace(inverse @ identity_matrix @ inverse @ dense_matrix)

    for factor in (dense_factor, structured_factor):
        np.testing.assert_allclose(
            factor.trace_inverse_penalty(identity),
            expected_identity_trace,
        )
        np.testing.assert_allclose(
            factor.trace_inverse_penalty(dense_penalty),
            expected_dense_trace,
        )
        np.testing.assert_allclose(
            factor.penalty_cross_trace(identity, identity, 2.0, 3.0),
            6.0 * expected_identity_self,
        )
        np.testing.assert_allclose(
            factor.penalty_cross_trace(identity, dense_penalty, 2.0, 3.0),
            6.0 * expected_cross,
        )


def test_compact_centered_and_low_rank_operator_products_match_dense():
    structured_factor, H = _contiguous_scalar_factor()
    inverse = np.linalg.inv(H)
    dense_factor = DenseHessianFactor(
        inverse=inverse,
        log_det=np.linalg.slogdet(H)[1],
    )
    rng = np.random.default_rng(818)
    base = SymmetricBlockOperator(
        A=rng.normal(size=(3, 3)),
        C=rng.normal(size=(4, 3)),
        d=rng.normal(size=4),
        small_indices=np.arange(3, dtype=np.intp),
        structured_indices=np.arange(3, 7, dtype=np.intp),
    )
    base = SymmetricBlockOperator(
        A=0.5 * (base.A + base.A.T),
        C=base.C,
        d=base.d,
        small_indices=base.small_indices,
        structured_indices=base.structured_indices,
    )
    centered = CenteredBlockOperator(
        raw=base,
        cross=rng.normal(size=7),
        total=-0.35,
        center=rng.normal(size=7),
    )
    low_rank = LowRankSymmetricOperator(
        basis=rng.normal(size=(7, 2)),
        core=np.array([[0.3, -0.2], [-0.2, 0.5]]),
    )
    combined = SumBlockOperator((centered, low_rank))
    other = SymmetricBlockOperator(
        A=np.diag([0.4, -0.1, 0.2]),
        C=rng.normal(scale=0.2, size=(4, 3)),
        d=rng.normal(scale=0.3, size=4),
        small_indices=np.arange(3, dtype=np.intp),
        structured_indices=np.arange(3, 7, dtype=np.intp),
    )
    combined_dense = materialize_compact_operator(combined)
    other_dense = materialize_compact_operator(other)
    identity = PenaltyComponent(
        name="broker",
        group_name="broker",
        group_index=1,
        group_sl=slice(3, 7),
        omega_raw=None,
        penalty_kind="identity",
    )
    identity_matrix = np.diag([0.0, 0.0, 0.0, 1.7, 1.7, 1.7, 1.7])

    for factor in (dense_factor, structured_factor):
        np.testing.assert_allclose(
            factor.trace_inverse_operator(combined),
            np.trace(inverse @ combined_dense),
            atol=2e-12,
        )
        inverse_product = inverse @ combined_dense
        np.testing.assert_allclose(
            factor.inverse_operator_diagonal(combined),
            np.diag(inverse_product),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            factor.inverse_operator_square_diagonal(combined),
            np.diag(inverse_product @ inverse_product),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            factor.operator_cross_trace(combined, other),
            np.trace(inverse @ combined_dense @ inverse @ other_dense),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            factor.penalty_operator_cross_trace(identity, 1.7, combined),
            np.trace(inverse @ identity_matrix @ inverse @ combined_dense),
            atol=2e-12,
        )


def _penalty(name, group_sl, omega=None, **kind):
    return PenaltyComponent(
        name=name,
        group_name=name,
        group_index=0,
        group_sl=group_sl,
        omega_raw=omega,
        omega_ssp=omega,
        **kind,
    )


def _symmetric(rng, size):
    values = rng.normal(size=(size, size))
    return values + values.T


def _scalar_directions(profiled: bool, operator_type=SymmetricBlockOperator):
    """A scalar Schur factor, raw or intercept-profiled, and three REML directions."""
    rng = np.random.default_rng(2609)
    q, K = 3, 4
    small_width = q + int(profiled)  # the augmented intercept joins the small block
    C = rng.normal(scale=0.4, size=(K, small_width))
    d = rng.uniform(1.0, 2.0, size=K)
    root = rng.normal(size=(small_width, small_width))
    A = root.T @ root + np.eye(small_width) + C.T @ (C / d[:, None])
    factor = ScalarSchurFactor(
        A=A,
        C=C,
        d=d,
        small_indices=np.arange(small_width),
        structured_indices=np.arange(small_width, small_width + K),
        term_name="re",
    )
    if profiled:
        factor = ProfiledScalarSchurFactor(
            augmented_factor=factor, sum_w=A[0, 0], xtw=np.concatenate([A[0, 1:], C[:, 0]])
        )

    def operator():
        return operator_type(
            A=_symmetric(rng, q),
            C=rng.normal(size=(K, q)),
            d=rng.normal(size=K),
            small_indices=factor.small_indices,
            structured_indices=factor.structured_indices,
        )

    width = q + K
    centered = CenteredBlockOperator(
        raw=operator(), cross=rng.normal(size=width), total=-0.35, center=rng.normal(size=width)
    )
    low_rank = LowRankSymmetricOperator(
        basis=rng.normal(size=(width, 2)), core=np.array([[0.3, -0.2], [-0.2, 0.5]])
    )
    return factor, [
        (_penalty("spline", slice(0, 2), np.array([[1.5, 0.2], [0.2, 0.8]])), 2.0, operator()),
        (
            _penalty("ridge", slice(2, 3), penalty_kind="identity"),
            0.7,
            SumBlockOperator((centered, low_rank)),
        ),
        (_penalty("re", slice(q, width), penalty_kind="identity"), 3.1, None),
    ]


def _block_directions(profiled: bool, operator_type=BlockSymmetricOperator):
    """A block Schur factor, raw or intercept-profiled, and three REML directions."""
    rng = np.random.default_rng(2610)
    n_levels, block_size, q = 4, 2, 3
    small_width = q + int(profiled)  # the augmented intercept joins the small block
    roots = rng.normal(size=(n_levels, block_size, block_size))
    D = np.einsum("kji,kjl->kil", roots, roots) + np.eye(block_size)
    C = rng.normal(scale=0.3, size=(n_levels, block_size, small_width))
    root = rng.normal(size=(small_width, small_width))
    A = root.T @ root + np.eye(small_width) + np.einsum("kiq,kir->qr", C, np.linalg.solve(D, C))
    factor = BlockSchurFactor(
        A=A,
        C=C,
        D=D,
        small_indices=np.arange(small_width),
        structured_indices=np.arange(small_width, small_width + n_levels * block_size).reshape(
            n_levels, block_size
        ),
        term_name="fs",
    )
    if profiled:
        factor = ProfiledBlockSchurFactor(
            augmented_factor=factor,
            sum_w=A[0, 0],
            xtw=np.concatenate([A[0, 1:], C[:, :, 0].ravel()]),
        )

    def operator():
        local = rng.normal(size=(n_levels, block_size, block_size))
        return operator_type(
            A=_symmetric(rng, q),
            C=rng.normal(size=(n_levels, block_size, q)),
            D=local + local.transpose(0, 2, 1),
            small_indices=factor.small_indices,
            structured_indices=factor.structured_indices,
        )

    width = q + n_levels * block_size
    repeated = _penalty(
        "fs",
        slice(q, width),
        np.diag([1.4, 0.0]),
        penalty_kind="repeated",
        repeat_count=n_levels,
        block_width=block_size,
    )
    centered = CenteredBlockOperator(
        raw=operator(), cross=rng.normal(size=width), total=1.1, center=rng.normal(size=width)
    )
    low_rank = LowRankSymmetricOperator(basis=rng.normal(size=(width, 2)), core=np.eye(2))
    return factor, [
        (repeated, 1.3, centered),
        (
            _penalty("small", slice(0, q), np.diag([0.4, 0.7, 1.0])),
            0.6,
            SumBlockOperator((operator(), low_rank)),
        ),
        (_penalty("ridge", slice(0, 2), penalty_kind="identity"), 2.2, None),
    ]


def _bare_block_directions():
    """A block Schur factor with no dense-small block, and a repeated penalty."""
    rng = np.random.default_rng(2611)
    n_levels, block_size = 4, 2
    width = n_levels * block_size
    roots = rng.normal(size=(n_levels, block_size, block_size))
    local = rng.normal(size=(n_levels, block_size, block_size))
    layout = {
        "small_indices": np.arange(0),
        "structured_indices": np.arange(width).reshape(n_levels, block_size),
    }
    factor = BlockSchurFactor(
        A=np.zeros((0, 0)),
        C=np.zeros((n_levels, block_size, 0)),
        D=np.einsum("kji,kjl->kil", roots, roots) + np.eye(block_size),
        term_name="fs",
        **layout,
    )
    operator = BlockSymmetricOperator(
        A=np.zeros((0, 0)),
        C=np.zeros((n_levels, block_size, 0)),
        D=local + local.transpose(0, 2, 1),
        **layout,
    )
    repeated = _penalty(
        "fs",
        slice(0, width),
        np.diag([1.4, 0.0]),
        penalty_kind="repeated",
        repeat_count=n_levels,
        block_width=block_size,
    )
    return factor, [
        (repeated, 1.3, operator),
        (_penalty("ridge", slice(0, width), penalty_kind="identity"), 0.4, None),
    ]


def _dense_direction(component, scale, operator, width):
    matrix = np.zeros((width, width))
    indices = np.arange(width)[component.group_sl]
    if component.penalty_kind == "identity":
        matrix[indices, indices] = scale
    else:
        matrix[np.ix_(indices, indices)] = scale * _component_omega(component, width)
    return matrix if operator is None else matrix + materialize_compact_operator(operator)


def _cross_trace_tolerance(factor, dense_directions):
    """First-order float64 bound on tr(Z O_i Z O_j) evaluated two ways.

    Every evaluation splits ``Z = Z_local + U R U'`` and accumulates inner
    products no longer than ``p + q``, so each is within
    ``gamma_{p+q} F_i F_j`` of the exact value, where
    ``F = (||Z_local|| + ||U||^2 ||R||) ||O||_F`` bounds the Frobenius norms
    of the split pieces (Higham 2002, eq. 3.13). The dense reference also
    forms ``Z`` and two products: five gammas in all.
    """
    if hasattr(factor, "_inverse_bdlr"):
        inverse = factor._inverse_bdlr()
        local = np.max(np.linalg.norm(inverse.blocks, 2, axis=(1, 2)))
    else:
        inverse = factor._inverse_dlr()
        local = np.max(np.abs(inverse.diagonal))
    low_rank = (
        np.linalg.norm(inverse.basis, 2) ** 2 * np.linalg.norm(inverse.core, 2)
        if inverse.core.size
        else 0.0
    )
    split = local + low_rank
    norms = split * np.array([np.linalg.norm(matrix) for matrix in dense_directions])
    n = factor.shape[0] + inverse.core.shape[0]
    gamma = n * np.finfo(np.float64).eps / (1.0 - n * np.finfo(np.float64).eps)
    return 5.0 * gamma * np.outer(norms, norms)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _scalar_directions(profiled=False),
        lambda: _scalar_directions(profiled=True),
        lambda: _block_directions(profiled=False),
        lambda: _block_directions(profiled=True),
        _bare_block_directions,
    ],
    ids=["scalar", "profiled-scalar", "block", "profiled-block", "block-without-small"],
)
def test_derivative_cross_traces_match_dense_reference(build):
    factor, directions = build()
    width = factor.shape[0]
    inverse = factor.solve(np.eye(width))
    dense = [_dense_direction(*direction, width) for direction in directions]
    expected = np.array(
        [[np.trace(inverse @ left @ inverse @ right) for right in dense] for left in dense]
    )

    traces = factor.derivative_cross_traces(directions)

    assert np.all(np.abs(traces - expected) <= _cross_trace_tolerance(factor, dense))


def test_scalar_derivative_cross_traces_refuse_a_foreign_block_layout():
    """The batched local trace drops each operator's small and cross blocks.

    That is exact only when the operator shares the factor's structured
    indices, so another layout is refused rather than traced wrongly.
    """
    factor, directions = _scalar_directions(profiled=False)
    component, scale, _ = directions[0]
    width = factor.shape[0]
    foreign = SymmetricBlockOperator(
        A=np.eye(width - 2),
        C=np.ones((2, width - 2)),
        d=np.ones(2),
        small_indices=np.arange(width - 2),
        structured_indices=np.arange(width - 2, width),
    )

    with pytest.raises(ValueError, match="structured block layout"):
        factor.derivative_cross_traces([(component, scale, foreign)])


@pytest.mark.parametrize(
    ("build", "operator_type", "multiply", "with_operators"),
    [
        (_scalar_directions, SymmetricBlockOperator, "_multiply_symmetric_dlr", True),
        (_block_directions, BlockSymmetricOperator, "_multiply_symmetric_bdlr", True),
        (_block_directions, BlockSymmetricOperator, "_multiply_symmetric_bdlr", False),
    ],
    ids=["scalar", "block", "block-penalty-only"],
)
def test_structured_reml_hessian_forms_each_inverse_product_once(
    monkeypatch, build, operator_type, multiply, with_operators
):
    """One ``H^-1 (lambda Omega + dH)`` product per direction per Hessian evaluation.

    A product is counted where it is formed: a direction applying its
    operator to the inverse basis, or a pairwise multiplication by the
    inverse's compact form. The pairwise path formed up to six such products
    for every pair of directions, and on block factors penalty-only pairs
    formed two.
    """
    created, formed = [], []

    class CountedOperator(operator_type):
        def __post_init__(self):
            super().__post_init__()
            created.append(id(self))

        def matvec(self, rhs):
            formed.append(id(self))
            return super().matvec(rhs)

    pairwise_multiply = getattr(factors_module, multiply)

    def counted_multiply(left, right):
        formed.append("pairwise")
        return pairwise_multiply(left, right)

    monkeypatch.setattr(factors_module, multiply, counted_multiply)
    factor, directions = build(profiled=True, operator_type=CountedOperator)
    if not with_operators:
        directions = [(component, scale, None) for component, scale, _ in directions]
        created.clear()
    penalties = [component for component, _, _ in directions]
    lambdas = {component.name: scale for component, scale, _ in directions}
    operators = {i: op for i, (_, _, op) in enumerate(directions) if op is not None}
    common = {"gradient": np.zeros(len(penalties)), "reml_penalties": penalties}

    hessian = reml_direct_hessian(
        [], Poisson(), factor, lambdas, dH_extra=operators or None, **common
    )

    assert Counter(formed) == Counter(created)
    width = factor.shape[0]
    dense_operators = {i: materialize_compact_operator(op) for i, op in operators.items()}
    dense_hessian = reml_direct_hessian(
        [],
        Poisson(),
        factor.solve(np.eye(width)),
        lambdas,
        dH_extra=dense_operators or None,
        **common,
    )
    dense_directions = [_dense_direction(*direction, width) for direction in directions]
    tolerance = 0.5 * _cross_trace_tolerance(factor, dense_directions)
    assert np.all(np.abs(hessian - dense_hessian) <= tolerance)


def test_centered_independent_blocks_certify_local_factor_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 129
    block_size = 4
    rng = np.random.default_rng(0)
    local_factor = rng.normal(size=(4, 3)) @ rng.normal(size=(3, block_size))
    local_factors = np.tile(local_factor, (n_levels, 1, 1))
    row_null = np.linalg.svd(local_factor.T, full_matrices=True)[0][:, -1]
    small = np.vstack(
        [
            np.column_stack(
                (
                    row_null * np.sin((level + 1) * 0.37),
                    row_null * np.cos((level + 1) * 0.23),
                )
            )
            for level in range(n_levels)
        ]
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("wide certification-limited FS inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    operator, public_design = _independent_centered_operator(local_factors, small)
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    preliminary = decompose_gram(operator.raw.D[0])

    assert decompose_factor(local_factor).rank == 3
    assert needs_factor_certification(preliminary)
    assert np.count_nonzero(expected.coefficient_estimable()) == small.shape[1]
    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_centered_independent_column_scale_certifies_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(0)
    local_factors = rng.normal(size=(2, 3, 2))
    small = (1e9 + np.arange(6, dtype=np.float64))[:, None]
    operator, public_design = _independent_centered_operator(local_factors, small)
    centered_design = public_design - np.mean(public_design, axis=0)
    expected = decompose_factor(centered_design).coefficient_estimable()

    assert np.all(expected)

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("cancellation-certified FS inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected,
    )


def test_centered_estimability_certification_failure_propagates_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.solvers._structured.geometry as structured_geometry

    operator = _unit_level_centered_operator(8)

    def fail_contract(_operator):
        raise ValueError("contract bug")

    monkeypatch.setattr(
        structured_geometry,
        "_independent_block_centered_estimability",
        fail_contract,
    )

    with pytest.raises(ValueError, match="contract bug"):
        centered_operator_coefficient_estimable(operator)


def test_centered_estimability_certification_failure_uses_bounded_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.solvers._structured.geometry as structured_geometry

    operator = _unit_level_centered_operator(128)
    expected = structured_geometry._bounded_centered_estimability(operator)

    def fail_numerically(_operator):
        raise np.linalg.LinAlgError("non-convergence")

    monkeypatch.setattr(
        structured_geometry,
        "_independent_block_centered_estimability",
        fail_numerically,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected,
    )


def test_centered_estimability_certification_failure_raises_when_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.solvers._structured.geometry as structured_geometry

    operator = _unit_level_centered_operator(513)

    def fail_numerically(_operator):
        raise np.linalg.LinAlgError("non-convergence")

    monkeypatch.setattr(
        structured_geometry,
        "_independent_block_centered_estimability",
        fail_numerically,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Compact structured estimability certification failed.*bounded dense fallback",
    ) as caught:
        centered_operator_coefficient_estimable(operator)
    assert isinstance(caught.value.__cause__, np.linalg.LinAlgError)


def test_centered_independent_schur_exact_alias_uses_factor_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 5
    block_size = 2
    rng = np.random.default_rng(0)
    local_factors = rng.normal(size=(n_levels, block_size, block_size))
    structured = np.zeros((n_levels * block_size, n_levels * block_size))
    for level, factor in enumerate(local_factors):
        block_slice = slice(level * block_size, (level + 1) * block_size)
        structured[block_slice, block_slice] = factor
    alias_map = rng.normal(size=(structured.shape[1], 3))
    operator, public_design = _independent_centered_operator(
        local_factors,
        structured @ alias_map,
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    assert not np.any(expected.coefficient_estimable()[: alias_map.shape[1]])

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("independent structured inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_centered_independent_scale_separated_aliases_preserve_null_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 4
    block_size = 2
    rng = np.random.default_rng(0)
    local_factors = rng.normal(size=(n_levels, block_size, block_size))
    local_factors *= np.array([1.0, 1e6])[None, None, :]
    structured = np.zeros((n_levels * block_size, n_levels * block_size))
    for level, factor in enumerate(local_factors):
        block_slice = slice(level * block_size, (level + 1) * block_size)
        structured[block_slice, block_slice] = factor
    alias_map = rng.normal(size=(structured.shape[1], 3))
    alias_map *= np.array([1.0, 1e-5, 1e-10])
    operator, public_design = _independent_centered_operator(
        local_factors,
        structured @ alias_map,
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    assert not np.any(expected.coefficient_estimable())

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("scale-separated FS inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_centered_independent_mixed_scaled_deficiency_preserves_independent_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 3
    rows_per_level = 4
    block_size = 4
    rng = np.random.default_rng(7)
    deficient = rng.normal(size=(rows_per_level, 2)) @ rng.normal(size=(2, 3))
    independent = rng.normal(size=rows_per_level)
    local_factor = np.column_stack((1e8 * independent, deficient))
    local_factors = np.tile(local_factor, (n_levels, 1, 1))
    operator, public_design = _independent_centered_operator(
        local_factors,
        np.empty((n_levels * rows_per_level, 0)),
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    independent_indices = np.arange(0, n_levels * block_size, block_size)
    np.testing.assert_array_equal(
        np.flatnonzero(expected.coefficient_estimable()),
        independent_indices,
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("mixed scaled FS inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_wide_centered_independent_multidimensional_scaled_null_preserves_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_levels = 74
    rows_per_level = 4
    block_size = 7
    rng = np.random.default_rng(1894)
    independent = rng.normal(size=rows_per_level)
    deficient = rng.normal(size=(rows_per_level, 2)) @ rng.normal(size=(2, block_size - 1))
    local_factor = np.column_stack((independent, deficient))
    local_factor *= np.geomspace(8e-6, 2e5, block_size)
    local_factors = np.tile(local_factor, (n_levels, 1, 1))
    operator, public_design = _independent_centered_operator(
        local_factors,
        np.empty((n_levels * rows_per_level, 0)),
    )
    expected = decompose_factor(public_design - np.mean(public_design, axis=0))
    independent_indices = np.arange(0, n_levels * block_size, block_size)
    np.testing.assert_array_equal(
        np.flatnonzero(expected.coefficient_estimable()),
        independent_indices,
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("wide multidimensional FS null inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )


def test_centered_independent_lifted_null_uses_design_column_scale(
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
    structured = np.zeros((6, 4))
    structured[:3, :2] = local_factor
    structured[3:, 2:] = local_factor
    small = (1e-7 * structured[:, 0] + structured[:, 1])[:, None]
    operator, public_design = _independent_centered_operator(local_factors, small)
    expected = decompose_factor(public_design)
    np.testing.assert_array_equal(
        expected.coefficient_estimable(),
        np.array([False, True, False, True, True]),
    )

    def reject_dense_fallback(_operator: CenteredBlockOperator) -> np.ndarray:
        raise AssertionError("equilibrated FS lifted-null inference must remain compact")

    monkeypatch.setattr(
        "superglm.solvers._structured.geometry._bounded_centered_estimability",
        reject_dense_fallback,
    )

    np.testing.assert_array_equal(
        centered_operator_coefficient_estimable(operator),
        expected.coefficient_estimable(),
    )
