"""Exact scalar Schur-factor algebra tests."""

import importlib.util
from dataclasses import FrozenInstanceError
from importlib import import_module

import numpy as np
import pytest

from superglm.solvers.hessian_factor import DenseHessianFactor, HessianFactor
from superglm.solvers.structured import ScalarSchurFactor, SymmetricBlockOperator
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
