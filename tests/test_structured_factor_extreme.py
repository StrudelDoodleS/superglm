"""Exact small systems at structured-factor arithmetic boundaries."""

import math

import numpy as np
import pytest

from superglm.group_matrix import DenseGroupMatrix, RandomEffectGroupMatrix
from superglm.solvers._structured.assembly import solve_cached_scalar_structured
from superglm.solvers._structured.factors import BlockSchurFactor, ScalarSchurFactor
from superglm.solvers._structured.moments import build_scalar_structured_system
from superglm.solvers._structured.selection import resolve_structured_backend
from superglm.types import GroupSlice


def _factor(kind, A, C, d):
    q = len(A)
    k = len(d)
    common = dict(A=A, small_indices=np.arange(q), term_name="dyadic")
    if kind == "scalar":
        return ScalarSchurFactor(**common, C=C, d=d, structured_indices=np.arange(q, q + k))
    return BlockSchurFactor(
        **common,
        C=C[:, None, :],
        D=d[:, None, None],
        structured_indices=np.arange(q, q + k)[:, None],
    )


@pytest.mark.parametrize("kind", ["scalar", "block"])
def test_large_retained_coupling_cannot_hide_a_schur_null_volume(kind):
    # H=R.T R, R=[[f,1,1],[f,0,0]], has rank 2 and pdet=2*f**2.
    # Schur elimination alone would publish f**2 and omit log(2).
    f = float(2**40)
    with pytest.raises(np.linalg.LinAlgError, match="coupled rank-deficient"):
        _factor(kind, np.array([[2 * f * f, f], [f, 1.0]]), np.array([[f, 1.0]]), np.ones(1))


def test_selected_positive_penalty_cannot_publish_a_rounded_alias_determinant():
    f = float(2**40)
    matrices = [
        DenseGroupMatrix(np.array([[0.0], [2 * f]])),
        RandomEffectGroupMatrix(np.array([0, 0]), 1),
    ]
    groups = [GroupSlice("x", 0, 1, penalized=False), GroupSlice("g", 1, 2)]
    weights = np.array([0.5, 0.5])
    lambdas = {"g": float.fromhex("0x1p-60")}
    decision = resolve_structured_backend(
        matrices,
        groups,
        direct_solve="structured",
        coefficient_width=2,
        row_weights=weights,
        lambda2=lambdas,
    )
    assert decision.use_structured
    system = build_scalar_structured_system(
        matrices, groups, weights, np.zeros(2), dominant_group_index=1
    )
    # The source PD determinant is lambda*f**2; the rounded PSD pdet is
    # 2*f**2. The cached Gram cannot certify the source determinant.
    with pytest.raises(np.linalg.LinAlgError, match="coupled rank-deficient"):
        solve_cached_scalar_structured(system, matrices, groups, lambdas)


@pytest.mark.parametrize("kind", ["scalar", "block"])
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("exponent", [0, -1030])
def test_finite_local_rhs_does_not_require_a_representable_inverse(kind, small, exponent):
    d = math.ldexp(1.0, exponent)
    q = int(small)
    factor = _factor(kind, np.eye(q), np.zeros((1, q)), np.array([d]))
    diagonal = np.concatenate((np.ones(q), [d]))
    expected = np.column_stack((np.ones(q + 1), np.full(q + 1, 2.0)))
    rhs = diagonal[:, None] * expected
    solution = factor.solve(rhs)
    assert factor.rank == q + 1
    assert not factor.rank_truncated
    assert np.all(np.isfinite(solution))
    np.testing.assert_allclose(solution, expected, rtol=16 * np.finfo(float).eps, atol=0)
    # Divide the represented diagonal equation before checking its residual,
    # so the assertion itself does not underflow an error at tiny d.
    np.testing.assert_allclose(
        solution, rhs / diagonal[:, None], rtol=16 * np.finfo(float).eps, atol=0
    )
    assert factor.logdet() == pytest.approx(
        exponent * math.log(2), abs=16 * np.finfo(float).eps * max(1, abs(exponent))
    )


@pytest.mark.parametrize("kind", ["scalar", "block"])
def test_tiny_local_cross_solve_preserves_finite_elimination(kind):
    d = float.fromhex("0x1p-1030")
    # [[1,d],[d,d]] is PD, and H@[0,1]=[d,d] exactly.
    factor = _factor(kind, np.ones((1, 1)), np.array([[d]]), np.array([d]))
    actual = factor.solve(np.array([d, d]))
    assert factor.rank == 2
    np.testing.assert_allclose(actual, [0.0, 1.0], rtol=16 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize("kind", ["scalar", "block"])
def test_unrepresentable_inverse_request_refuses_after_a_finite_solve(kind):
    d = float.fromhex("0x1p-1030")
    factor = _factor(kind, np.ones((1, 1)), np.zeros((1, 1)), np.array([d]))
    np.testing.assert_allclose(
        factor.solve(np.array([1.0, d])), [1.0, 1.0], rtol=16 * np.finfo(float).eps, atol=0
    )
    np.testing.assert_array_equal(factor.selected_inverse_diagonal(np.array([0])), [1.0])
    with pytest.raises(np.linalg.LinAlgError, match="inverse"):
        factor.selected_inverse_diagonal(np.array([1]))
    with pytest.raises(np.linalg.LinAlgError, match="inverse"):
        factor.selected_inverse_block(np.array([1]))


@pytest.mark.parametrize("kind", ["scalar", "block"])
def test_schur_fallback_does_not_turn_negative_curvature_into_positive_rank(kind):
    with pytest.raises(np.linalg.LinAlgError, match="negative.*Schur|Schur.*negative"):
        _factor(kind, -np.ones((1, 1)), np.zeros((1, 1)), np.ones(1))


@pytest.mark.parametrize("kind", ["scalar", "block"])
@pytest.mark.parametrize("coupling", [0.0, 2.0**-30])
def test_roundoff_sized_null_volume_remains_admissible(kind, coupling):
    f = float(2**40)
    factor = _factor(
        kind,
        np.array([[2 * f * f, f * coupling], [f * coupling, coupling**2]]),
        np.array([[f, coupling]]),
        np.ones(1),
    )
    expected = 80 * math.log(2) + math.log1p(coupling**2)
    assert factor.rank == 2
    assert factor.logdet() == pytest.approx(expected, abs=16 * np.finfo(float).eps * expected)


@pytest.mark.parametrize("integer_block", [[[2, 1], [1, 2]], [[3, 1], [1, 2]]])
@pytest.mark.parametrize("small", [False, True])
def test_subnormal_dense_local_block_preserves_normalized_solve_and_volume(integer_block, small):
    eta = math.ldexp(1.0, -1074)
    block = np.asarray(integer_block, dtype=float)
    q = int(small)
    factor = BlockSchurFactor(
        A=np.eye(q),
        C=np.zeros((1, 2, q)),
        D=(eta * block)[None, :, :],
        small_indices=np.arange(q),
        structured_indices=np.arange(q, q + 2)[None, :],
        term_name="subnormal_block",
    )
    expected = np.array([1.0, -1.0])
    normalized_rhs = block @ expected
    solution = factor.solve(np.concatenate((np.zeros(q), eta * normalized_rhs)))
    assert factor.rank == q + 2
    np.testing.assert_allclose(solution[q:], expected, rtol=32 * np.finfo(float).eps, atol=0)
    residual = block @ solution[q:] - normalized_rhs
    scale = np.linalg.norm(block, ord=np.inf) * np.linalg.norm(solution[q:], ord=np.inf)
    np.testing.assert_allclose(residual, 0, rtol=0, atol=32 * np.finfo(float).eps * scale)
    determinant = integer_block[0][0] * integer_block[1][1] - 1
    expected_logdet = -2148 * math.log(2) + math.log(determinant)
    assert factor.logdet() == pytest.approx(
        expected_logdet, abs=16 * np.finfo(float).eps * abs(expected_logdet)
    )


def test_exceptional_local_scaling_preserves_a_normal_and_subnormal_coordinate_together():
    eta = math.ldexp(1.0, -1074)
    diagonal = np.array([1.0, eta])
    factor = BlockSchurFactor(
        A=np.empty((0, 0)),
        C=np.empty((1, 2, 0)),
        D=np.diag(diagonal)[None, :, :],
        small_indices=np.array([], dtype=int),
        structured_indices=np.array([[0, 1]]),
        term_name="mixed_coordinate_scale",
    )
    solution = factor.solve(np.diag(diagonal))
    assert factor.rank == 2
    np.testing.assert_allclose(solution, np.eye(2), rtol=0, atol=16 * np.finfo(float).eps)
    assert factor.logdet() == pytest.approx(
        -1074 * math.log(2), abs=16 * np.finfo(float).eps * 1074
    )
