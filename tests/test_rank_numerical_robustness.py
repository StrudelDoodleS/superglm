"""Focused regressions for shared rank-decomposition arithmetic and authority."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    _exact_orthogonal_columns,
    _exact_to_float,
    _exact_triangular_solve,
    _symmetric_part,
    decompose_factor,
    decompose_gram,
    decompose_gram_if_authoritative,
    decompose_symmetric,
    needs_factor_certification,
)


@pytest.mark.parametrize("exponent", [-1074, -1050, -1000, -60, 0, 60, 1023])
def test_negative_diagonal_refusal_is_invariant_to_objective_units(exponent: int) -> None:
    """A negative Rayleigh quotient remains negative under positive rescaling."""
    scale = np.ldexp(1.0, exponent)
    matrix = np.diag([scale, -scale])

    with pytest.raises(ValueError, match="materially negative diagonal"):
        decompose_gram(matrix)

    signed = decompose_symmetric(matrix)
    assert signed.rank == 2
    assert np.isfinite(signed.log_pdet)


@pytest.mark.parametrize("exponent", [-1074, -1050, -1000, -60, 0, 60, 1023])
def test_zero_diagonal_psd_control_keeps_its_rank_across_objective_units(exponent: int) -> None:
    scale = np.ldexp(1.0, exponent)
    decomposition = decompose_gram(np.diag([scale, 0.0]))

    assert decomposition.rank == 1
    expected = exponent * np.log(2.0)
    tolerance = 8 * np.finfo(float).eps * max(abs(expected), 1.0)
    assert abs(decomposition.log_pdet - expected) <= tolerance


@pytest.mark.parametrize("exponent", [-900, -60, 0, 60, 900])
def test_diagonal_roundoff_allowance_stays_relative_to_matrix_scale(exponent: int) -> None:
    """Keep the existing relative PSD allowance while removing its unit floor."""
    scale = np.ldexp(1.0, exponent)
    matrix = np.diag([scale, -np.finfo(float).eps * scale])

    assert decompose_gram(matrix).rank == 1


def test_structural_zero_padding_preserves_factor_certification_authority() -> None:
    factor = np.array([[1.0, 1.0], [0.0, 1e-7]])
    gram = factor.T @ factor
    padded = np.pad(gram, ((0, 1), (0, 1)))

    certified_factor = decompose_factor(factor)
    assert certified_factor.rank == 2

    for matrix in (gram, padded):
        decomposition = decompose_gram(matrix)
        assert decomposition.rank == 2
        certification_condition = SHARED_RANK_POLICY.warning_condition / np.sqrt(
            SHARED_RANK_POLICY.certification_band
        )
        assert decomposition.pre_truncation_condition >= certification_condition
        assert needs_factor_certification(decomposition)
        assert decompose_gram_if_authoritative(matrix) is None


@pytest.mark.parametrize("scale", [1.0, 1e-170, 1e-309, np.ldexp(1.0, -1024), 1e160])
def test_factor_rhs_retains_finite_geometry_at_extreme_scale(scale: float) -> None:
    factor = scale * np.eye(2)
    target = np.array([1.0, -2.0])
    response = factor @ target

    decomposition = decompose_factor(factor, retain_factor_solve=True)
    coefficients = decomposition.solve_factor_rhs(response)
    arithmetic_tolerance = 128.0 * np.finfo(float).eps * max(factor.shape)

    assert decomposition.rank == 2
    assert np.all(np.isfinite(coefficients))
    np.testing.assert_allclose(
        coefficients,
        target,
        rtol=arithmetic_tolerance,
        atol=arithmetic_tolerance,
    )
    assert np.isfinite(decomposition.log_pdet)
    expected_log_pdet = 4.0 * np.log(scale)
    log_tolerance = arithmetic_tolerance * max(1.0, abs(expected_log_pdet))
    assert abs(decomposition.log_pdet - expected_log_pdet) <= log_tolerance
    np.testing.assert_allclose(
        factor @ coefficients / scale,
        target,
        rtol=arithmetic_tolerance,
        atol=arithmetic_tolerance,
    )


@pytest.mark.parametrize(
    "decomposition",
    [
        decompose_gram(np.diag([1e-308, 1.0])),
        decompose_factor(np.diag([1e-154, 1.0])),
    ],
)
def test_pseudo_inverse_agrees_with_solve_and_reconstructs_representable_gram(
    decomposition,
) -> None:
    gram = np.diag([1e-308, 1.0])
    inverse = decomposition.pseudo_inverse()
    arithmetic_tolerance = 128.0 * np.finfo(float).eps * 2

    assert np.all(np.isfinite(inverse))
    np.testing.assert_allclose(
        gram @ inverse @ gram,
        gram,
        rtol=arithmetic_tolerance,
        atol=0.0,
    )
    np.testing.assert_allclose(
        inverse @ np.array([1.0, 0.0]),
        decomposition.solve(np.array([1.0, 0.0])),
        rtol=arithmetic_tolerance,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.block(
            [
                [np.array([[1.0]]), np.zeros((1, 2))],
                [np.zeros((2, 1)), np.array([[0.0, 1.0], [1.0, 0.0]])],
            ]
        ),
    ],
)
def test_psd_route_refuses_nonzero_offdiagonal_zero_diagonal_rows(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="zero diagonal"):
        decompose_gram(matrix)
    with pytest.raises(ValueError, match="zero diagonal"):
        decompose_gram_if_authoritative(matrix)

    signed = decompose_symmetric(matrix)
    assert signed.rank == matrix.shape[0]
    assert np.count_nonzero(signed.retained_values < 0.0) == 1
    assert np.count_nonzero(signed.retained_values > 0.0) == matrix.shape[0] - 1
    bound = 128.0 * matrix.shape[0] * np.finfo(float).eps
    np.testing.assert_allclose(
        matrix @ signed.pseudo_inverse(), np.eye(matrix.shape[0]), atol=bound, rtol=bound
    )


@pytest.mark.parametrize("global_scale", [1.0, 1e-308, 1e308])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_signed_solve_preserves_permuted_and_padded_modes(
    global_scale: float, padding: int, reverse: bool
) -> None:
    # The active matrix squares to I, so its condition number is exactly one.
    # An epsilon-scaled zero diagonal loses its independent positive mode.
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    matrix = np.pad(global_scale * base, ((0, padding), (0, padding)))
    target = np.pad(np.array([0.25, -0.5, 0.75]), (0, padding))
    permutation = np.arange(len(matrix))
    if reverse:
        permutation = permutation[::-1]
    matrix = matrix[np.ix_(permutation, permutation)]
    target = target[permutation]
    active = permutation < 3
    response = matrix @ target

    decomposition = decompose_symmetric(matrix)
    bound = 128.0 * len(matrix) * np.finfo(float).eps
    assert decomposition.rank == 3
    assert np.count_nonzero(decomposition.retained_values > 0.0) == 2
    assert np.count_nonzero(decomposition.retained_values < 0.0) == 1
    coefficients = decomposition.solve(response)
    np.testing.assert_allclose(coefficients, target, atol=bound, rtol=bound)
    np.testing.assert_allclose(
        (matrix @ coefficients) / global_scale,
        response / global_scale,
        atol=bound,
        rtol=bound,
    )
    inverse = decomposition.pseudo_inverse()
    assert np.all(np.isfinite(inverse))
    np.testing.assert_allclose(
        matrix @ inverse, np.diag(active.astype(float)), atol=bound, rtol=bound
    )
    np.testing.assert_array_equal(decomposition.null_basis(), np.eye(len(matrix))[:, ~active])
    np.testing.assert_array_equal(decomposition.retained_parameter_basis()[~active], 0.0)


@pytest.mark.parametrize(
    "coordinate_scale",
    [
        np.array([0.5, 2.0, 0.5]),
        np.array([1e154, 1e-154, 1e-154]),
        np.array([1e-154, 1e154, 1e154]),
    ],
)
def test_signed_reconstruction_preserves_admissibly_rescaled_blocks(
    coordinate_scale: np.ndarray,
) -> None:
    # Each disconnected block remains a scalar multiple of an orthogonal
    # matrix. Check its inverse action, not coefficients of the ill-scaled
    # complete matrix. All nonzero products in A @ A_inverse have unit scale.
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    matrix = (coordinate_scale[:, None] * base) * coordinate_scale[None, :]
    decomposition = decompose_symmetric(matrix)

    bound = 128.0 * len(matrix) * np.finfo(float).eps
    assert decomposition.rank == 3
    assert np.count_nonzero(decomposition.retained_values > 0.0) == 2
    assert np.count_nonzero(decomposition.retained_values < 0.0) == 1
    inverse = decomposition.pseudo_inverse()
    assert np.all(np.isfinite(inverse))
    np.testing.assert_allclose(matrix @ inverse, np.eye(3), atol=bound, rtol=bound)
    expected_log_pdet = 2.0 * float(np.sum(np.log(coordinate_scale)))
    assert abs(decomposition.log_pdet - expected_log_pdet) <= bound * max(
        1.0, abs(expected_log_pdet)
    )


def test_signed_spectral_and_structural_null_spaces_keep_disjoint_row_supports() -> None:
    # The active block has eigenvalues -sqrt(6), 0, sqrt(6), with exact null
    # vector [1, -2, -1]. Its nonzero singular values are well conditioned.
    base = np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, -2.0]])
    permutation = np.array([3, 1, 4, 0, 2])
    matrix = np.pad(base, ((0, 2), (0, 2)))[np.ix_(permutation, permutation)]
    active = permutation < 3
    exact_null = np.array([1.0, -2.0, -1.0, 0.0, 0.0])[permutation]
    exact_null /= np.linalg.norm(exact_null)

    decomposition = decompose_symmetric(matrix)
    bound = 128.0 * len(matrix) * np.finfo(float).eps
    assert decomposition.rank == 2
    null = decomposition.null_basis()
    assert null.shape == (5, 3)
    np.testing.assert_array_equal(null[~active, :1], 0.0)
    np.testing.assert_array_equal(null[:, 1:], np.eye(5)[:, ~active])
    spectral_null = null[:, 0] / np.linalg.norm(null[:, 0])
    np.testing.assert_allclose(
        np.outer(spectral_null, spectral_null),
        np.outer(exact_null, exact_null),
        atol=bound,
        rtol=bound,
    )
    np.testing.assert_array_equal(decomposition.retained_parameter_basis()[~active], 0.0)
    np.testing.assert_allclose(
        matrix @ decomposition.pseudo_inverse() @ matrix,
        matrix,
        atol=bound * np.linalg.norm(matrix, ord=2),
        rtol=bound,
    )


def test_signed_subnormal_scaling_preserves_representable_log_determinant() -> None:
    # det(A) = -smallest**2. Its logarithm is representable even though its
    # inverse is not. Forming sqrt(3*smallest) * sqrt(smallest) first rounds
    # that denominator to 2*smallest and changes the retained determinant.
    smallest = np.nextafter(0.0, 1.0)
    matrix = np.array([[3.0 * smallest, smallest], [smallest, 0.0]])
    decomposition = decompose_symmetric(matrix)

    assert decomposition.rank == 2
    assert np.count_nonzero(decomposition.retained_values < 0.0) == 1
    expected_log_pdet = 2.0 * np.log(smallest)
    bound = 128.0 * len(matrix) * np.finfo(float).eps * abs(expected_log_pdet)
    assert abs(decomposition.log_pdet - expected_log_pdet) <= bound


def test_factor_refuses_unrepresentable_column_norm() -> None:
    # The exact norm is sqrt(2) * max_float, beyond the finite output range.
    factor = np.full((2, 1), np.finfo(float).max)
    with pytest.raises(ValueError, match="column norm is not representable"):
        decompose_factor(factor)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("coupling_sign", [-1.0, 1.0])
def test_signed_scaling_preserves_solve_relevant_subnormal_coupling(
    reverse: bool, coupling_sign: float
) -> None:
    matrix = np.array([[1e308, coupling_sign * 1e-308], [coupling_sign * 1e-308, 1e-308]])
    rhs = np.array([0.0, 1.0])
    if reverse:
        matrix = matrix[::-1, ::-1]
        rhs = rhs[::-1]

    decomposition = decompose_symmetric(matrix)
    coefficients = decomposition.solve(rhs)
    assert decomposition.rank == 2
    assert np.all(np.isfinite(coefficients))
    # Both products in the first row have unit magnitude. Losing half the
    # tiny coupling gives componentwise backward error 1/3, despite a finite
    # solution. Coefficient-forward accuracy is not asserted on this matrix.
    residual = np.abs(matrix @ coefficients - rhs)
    component_scale = np.abs(matrix) @ np.abs(coefficients) + np.abs(rhs)
    bound = 128.0 * len(matrix) * np.finfo(float).eps
    assert np.max(residual / component_scale) <= bound


@pytest.mark.parametrize("scale", [1e-309, 1e-320])
@pytest.mark.parametrize("padding", [0, 1])
def test_subnormal_factor_rhs_solves_the_represented_rotated_system(
    scale: float, padding: int
) -> None:
    active_factor = scale * np.array([[0.6, 0.8], [-0.8, 0.6]])
    response = active_factor @ np.array([1.0, -2.0])
    factor = np.pad(active_factor, ((0, 0), (0, padding)))
    # The represented rotated matrix still has condition number one. Its
    # rounded RHS need not correspond exactly to the generating coefficients.
    # Solve the represented inputs in normal units for the independent oracle.
    normalized_factor = active_factor / scale
    normalized_rhs = response / scale
    expected = np.linalg.solve(normalized_factor, normalized_rhs)

    with np.errstate(over="raise", invalid="raise"):
        decomposition = decompose_factor(factor, retain_factor_solve=True)
        actual = decomposition.solve_factor_rhs(response)
    bound = 128.0 * max(factor.shape) * np.finfo(float).eps
    assert decomposition.rank == 2
    np.testing.assert_allclose(actual, np.pad(expected, (0, padding)), atol=bound, rtol=bound)
    np.testing.assert_allclose(
        normalized_factor @ actual[:2], normalized_rhs, atol=bound, rtol=bound
    )
    np.testing.assert_array_equal(decomposition.null_basis(), np.eye(2 + padding)[:, 2:])


@pytest.mark.parametrize("retain_factor_solve", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
def test_tiny_factor_decomposition_preserves_representable_gram_rhs_solve(
    retain_factor_solve: bool,
    aliased: bool,
) -> None:
    # X = 2^-1028 I and rhs = 2^-1068 [1, -2], so the exact Gram solution
    # is 2^988 [1, -2]. X.T @ X and X^-1 are not representable intermediates.
    factor = np.ldexp(np.eye(2), -1028)
    rhs = np.ldexp(np.array([1.0, -2.0]), -1068)
    expected = np.ldexp(np.array([1.0, -2.0]), 988)
    if aliased:
        factor = np.ldexp(np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0]]), -1028)
        rhs = np.ldexp(np.array([1.0, -2.0, 2.0, 0.0]), -1068)
        expected = np.ldexp(np.array([1.0, -2.0, 0.0, 0.0]), 988)
    with np.errstate(over="raise", invalid="raise"):
        decomposition = decompose_factor(factor, retain_factor_solve=retain_factor_solve)
        actual = decomposition.solve(rhs)
    bound = 128.0 * len(rhs) * np.finfo(float).eps
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=bound)


@pytest.mark.parametrize(
    "operation", ["solve_factor_rhs", "pseudo_inverse", "retained_parameter_basis"]
)
def test_tiny_factor_refuses_an_unrepresentable_requested_result(operation: str) -> None:
    decomposition = decompose_factor(1e-309 * np.eye(2), retain_factor_solve=True)
    with pytest.raises(ValueError, match="not representable"):
        if operation == "solve_factor_rhs":
            decomposition.solve_factor_rhs(np.ones(2))
        else:
            getattr(decomposition, operation)()


@pytest.mark.parametrize("scale", [1e-309, 1e-320])
def test_tiny_deficient_factor_keeps_finite_null_geometry_and_estimability(scale: float) -> None:
    factor = scale * np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    response = factor @ np.array([1.0, -2.0, 0.0, 0.0])
    with np.errstate(over="raise", invalid="raise"):
        decomposition = decompose_factor(factor, retain_factor_solve=True)
        actual = decomposition.solve_factor_rhs(response)
        estimable = decomposition.coefficient_estimable()
        assert decomposition.is_estimable(np.array([1.0, 0.0, 2.0, 0.0]))
        assert decomposition.is_estimable(np.array([0.0, 1.0, 0.0, 0.0]))
        assert not decomposition.is_estimable(np.array([1.0, 0.0, 0.0, 0.0]))
        assert not decomposition.is_estimable(np.array([0.0, 0.0, 1.0, 0.0]))
        assert not decomposition.is_estimable(np.array([0.0, 0.0, 0.0, 1.0]))
    bound = 128.0 * max(factor.shape) * np.finfo(float).eps
    assert decomposition.rank == 2
    np.testing.assert_array_equal(estimable, [False, True, False, False])
    np.testing.assert_allclose(actual, [1.0, -2.0, 0.0, 0.0], atol=bound, rtol=bound)
    null = decomposition.null_basis()
    assert np.all(np.isfinite(null))
    np.testing.assert_array_equal(null[3, :1], 0.0)
    np.testing.assert_array_equal(null[:, 1], [0.0, 0.0, 0.0, 1.0])
    exact_null = np.array([-2.0, 0.0, 1.0, 0.0]) / np.sqrt(5.0)
    spectral_null = null[:, 0] / np.linalg.norm(null[:, 0])
    np.testing.assert_allclose(
        np.outer(spectral_null, spectral_null),
        np.outer(exact_null, exact_null),
        atol=bound,
        rtol=bound,
    )
    expected_log_pdet = np.log(5.0) + 4.0 * np.log(scale)
    assert abs(decomposition.log_pdet - expected_log_pdet) <= bound * abs(expected_log_pdet)


def test_tiny_mixed_scale_alias_preserves_estimable_functionals() -> None:
    factor = np.array([[1e-309, 0.0, 1.0], [0.0, 1.0, 0.0]])
    with np.errstate(over="raise", invalid="raise"):
        decomposition = decompose_factor(factor, retain_factor_solve=True)
        np.testing.assert_array_equal(decomposition.coefficient_estimable(), [False, True, False])
        assert decomposition.is_estimable(np.array([1e-309, 0.0, 1.0]))
        assert decomposition.is_estimable(np.array([0.0, 1.0, 0.0]))
        assert not decomposition.is_estimable(np.array([1.0, 0.0, 0.0]))
        assert not decomposition.is_estimable(np.array([0.0, 0.0, 1.0]))
    assert np.all(np.isfinite(decomposition.null_basis()))
    bound = 128.0 * max(factor.shape) * np.finfo(float).eps
    assert abs(decomposition.log_pdet) <= bound


def test_tiny_discarded_alias_keeps_representable_inverse_and_retained_basis() -> None:
    factor = np.array([[1.0, 0.0, 1e-309], [0.0, 1.0, 0.0]])
    with np.errstate(over="raise", invalid="raise"):
        decomposition = decompose_factor(factor, retain_factor_solve=True)
        inverse = decomposition.pseudo_inverse()
        retained = decomposition.retained_parameter_basis()
    # The selected columns are an orthonormal basis. Their inverse and
    # retained parameter basis remain representable despite the tiny alias.
    bound = 128.0 * max(factor.shape) * np.finfo(float).eps
    np.testing.assert_array_equal(decomposition.active_columns, [0, 1])
    np.testing.assert_allclose(inverse, np.diag([1.0, 1.0, 0.0]), atol=bound, rtol=bound)
    np.testing.assert_allclose(retained, np.eye(3)[:, :2], atol=bound, rtol=bound)


def test_smallest_factor_retains_rank_one_log_volume_and_null_space() -> None:
    smallest = np.nextafter(0.0, 1.0)
    row = np.array([1.0, 2.0, 3.0, 4.0])
    factor = smallest * row[None, :]
    with np.errstate(over="raise", invalid="raise"):
        decomposition = decompose_factor(factor, retain_factor_solve=True)
        actual = decomposition.solve_factor_rhs(np.array([2.0 * smallest]))
        assert decomposition.is_estimable(row)
        np.testing.assert_array_equal(
            decomposition.coefficient_estimable(), np.zeros(4, dtype=bool)
        )
    bound = 128.0 * factor.shape[1] * np.finfo(float).eps
    assert decomposition.rank == 1
    np.testing.assert_allclose(actual, [2.0, 0.0, 0.0, 0.0], atol=bound, rtol=bound)
    expected_log_pdet = np.log(30.0) + 2.0 * np.log(smallest)
    assert abs(decomposition.log_pdet - expected_log_pdet) <= bound * abs(expected_log_pdet)
    null = decomposition.null_basis()
    assert np.all(np.isfinite(null))
    orthogonal_null, _ = np.linalg.qr(null, mode="reduced")
    np.testing.assert_allclose(
        orthogonal_null @ orthogonal_null.T,
        np.eye(4) - np.outer(row, row) / 30.0,
        atol=bound,
        rtol=bound,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_tiny_discarded_alias_preserves_widely_separated_rhs_components(reverse: bool) -> None:
    factor = np.array([[1.0, 0.0, 1e-309], [0.0, 1.0, 0.0]])
    rhs = np.array([1e-300, 1e300, 0.0])
    if reverse:
        rhs = rhs[[1, 0, 2]]

    decomposition = decompose_factor(factor)
    actual = decomposition.solve(rhs)
    np.testing.assert_array_equal(decomposition.active_columns, [0, 1])
    # The selected Gram block is exactly I. A common RHS exponent erases
    # its small component even though the selected problem has condition one.
    bound = 128.0 * factor.shape[1] * np.finfo(float).eps
    np.testing.assert_allclose(actual, rhs, rtol=bound, atol=0.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_tiny_diagonal_solve_has_small_represented_componentwise_backward_error(
    reverse: bool,
) -> None:
    diagonal = np.array([1e-309, 1.0])
    rhs = np.array([np.nextafter(0.0, 1.0), 1e308])
    if reverse:
        diagonal = diagonal[::-1]
        rhs = rhs[::-1]
    decomposition = decompose_factor(np.diag(diagonal))
    actual = decomposition.solve(rhs)
    assert np.all(np.isfinite(actual))
    bound = 128.0 * len(rhs) * np.finfo(float).eps
    # Decimal uses the exact represented inputs and an extended exponent
    # range. Forming diagonal**2 in binary64 would erase the first equation.
    # Eighty decimal digits leave oracle rounding far below binary64 epsilon.
    with localcontext() as context:
        context.prec = 80
        for entry, response, coefficient in zip(diagonal, rhs, actual, strict=True):
            gram = Decimal.from_float(float(entry)) ** 2
            predicted = gram * Decimal.from_float(float(coefficient))
            target = Decimal.from_float(float(response))
            backward_error = abs(predicted - target) / (abs(predicted) + abs(target))
            assert backward_error <= Decimal.from_float(bound)


@pytest.mark.parametrize("coordinate_order", list(permutations(range(3))))
@pytest.mark.parametrize("padding", [0, 2])
def test_tiny_factor_null_span_preserves_two_independent_directions(
    coordinate_order: tuple[int, ...], padding: int
) -> None:
    factor = np.ldexp(np.ones((1, 3)), np.array([-1030, 60, 60]))
    factor = np.pad(factor, ((0, 0), (0, padding)))
    permutation = np.array(coordinate_order)
    if padding:
        permutation = np.array([3, coordinate_order[0], 4, *coordinate_order[1:]])
    factor = factor[:, permutation]
    active = permutation < 3
    witness = np.pad(np.array([0.0, 1.0, -1.0]), (0, padding))[permutation] / np.sqrt(2.0)

    decomposition = decompose_factor(factor)
    assert decomposition.rank == 1
    null = decomposition.null_basis()
    assert null.shape == (3 + padding, 2 + padding)
    assert np.all(np.isfinite(null))
    np.testing.assert_array_equal(null[~active, :2], 0.0)
    np.testing.assert_array_equal(null[:, 2:], np.eye(3 + padding)[:, ~active])
    bound = 128.0 * factor.shape[1] * np.finfo(float).eps
    singular_values = np.linalg.svd(null, compute_uv=False)
    assert np.count_nonzero(singular_values > bound * singular_values[0]) == 2 + padding
    # This exact witness must remain in the stored span. Finite columns with
    # correct row support alone do not establish that the span survived.
    np.testing.assert_array_equal(factor @ witness, 0.0)
    projection = null @ np.linalg.lstsq(null, witness, rcond=None)[0]
    assert np.linalg.norm(projection - witness) <= bound


def test_tiny_factor_complementary_log_volume_keeps_independent_null_directions() -> None:
    factor = np.zeros((3, 5))
    factor[0, :3] = np.ldexp(np.ones(3), [-1030, 60, 60])
    factor[1, 3] = 1.0
    factor[2, 4] = 1.0
    decomposition = decompose_factor(factor)

    assert decomposition.rank == 3
    # The three nonzero Gram eigenvalues are 2**121 + 2**-2060, 1, 1.
    # log1p(2**-2181) is below the stated logarithmic arithmetic allowance.
    expected = 121.0 * np.log(2.0)
    bound = 128.0 * factor.shape[1] * np.finfo(float).eps * abs(expected)
    assert abs(decomposition.log_pdet - expected) <= bound


@pytest.mark.parametrize("lower", [False, True])
def test_exact_triangular_substitution_preserves_cancellation_for_matrix_rhs(lower: bool) -> None:
    large = Fraction(2**600)
    small = Fraction(1, 2**700)
    upper = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    rhs = np.array([[large + small, -large - small], [large, -large], [small, -small]])
    expected = np.array([[small, -small], [large, -large], [small, -small]])
    if lower:
        upper = upper[::-1, ::-1]
        rhs = rhs[::-1]
        expected = expected[::-1]

    actual = _exact_triangular_solve(upper, rhs, lower=lower)
    np.testing.assert_array_equal(actual, expected)


def test_exact_result_conversion_rounds_subnormals_and_refuses_excess_magnitude() -> None:
    smallest = np.nextafter(0.0, 1.0)
    quarter = Fraction.from_float(smallest) / 4
    actual = _exact_to_float(np.array([quarter, 3 * quarter, -3 * quarter], dtype=object))
    np.testing.assert_array_equal(actual, [0.0, smallest, -smallest])
    maximum = Fraction.from_float(float(np.finfo(float).max))
    np.testing.assert_array_equal(
        _exact_to_float(np.array([maximum, -maximum], dtype=object)),
        [np.finfo(float).max, -np.finfo(float).max],
    )
    # maximum + 1 would round back to maximum. The explicit magnitude
    # check must still refuse a requested value outside the finite range.
    for value in (maximum + 1, -maximum - 1):
        with pytest.raises(ValueError, match="not representable"):
            _exact_to_float(np.array([value], dtype=object))


def test_exact_orthogonalization_preserves_span_and_log_volume_before_rounding() -> None:
    magnitude = Fraction(2**1500)
    coordinates = np.array([[magnitude, magnitude], [Fraction(1), Fraction(0)], [0, 1]])
    basis, log_volume = _exact_orthogonal_columns(coordinates)

    bound = 128.0 * len(coordinates) * np.finfo(float).eps
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=bound, rtol=bound)
    expected_projector = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, -0.5], [0.0, -0.5, 0.5]])
    # The omitted first coordinate of the exact orthogonal complement is
    # O(2**-1500). The Gram determinant is exactly 2**3001 + 1.
    np.testing.assert_allclose(basis @ basis.T, expected_projector, atol=bound, rtol=bound)
    expected_log_volume = 3001.0 * np.log(2.0)
    assert abs(log_volume - expected_log_volume) <= bound * abs(expected_log_volume)


@pytest.mark.parametrize("subnormal_units", [1, 3, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_extreme_symmetric_gram_preserves_each_resolved_direction(
    subnormal_units: int, reverse: bool
) -> None:
    diagonal = np.array([np.ldexp(1.0, 1023), subnormal_units * np.nextafter(0.0, 1.0)])
    if reverse:
        diagonal = diagonal[::-1]
    gram = np.diag(diagonal)

    # Averaging an already symmetric represented matrix is the identity,
    # including its subnormal coordinates next to a near-maximum coordinate.
    np.testing.assert_array_equal(_symmetric_part(gram), gram)
    decomposition = decompose_gram(gram)
    assert decomposition.rank == 2
    logarithms = np.log(diagonal)
    expected = float(np.sum(logarithms))
    bound = 8.0 * len(diagonal) * np.finfo(float).eps * float(np.sum(np.abs(logarithms)))
    assert abs(decomposition.log_pdet - expected) <= bound


def test_mixed_extreme_symmetric_average_matches_exact_entrywise_oracle() -> None:
    large = np.ldexp(1.0, 1023)
    tiny = np.nextafter(0.0, 1.0)
    values = np.array(
        [[large, large, tiny], [np.finfo(float).max, -large, 3 * tiny], [3 * tiny, 7 * tiny, tiny]]
    )
    expected = np.empty_like(values)
    for i, j in np.ndindex(values.shape):
        expected[i, j] = float(
            (Fraction.from_float(values[i, j]) + Fraction.from_float(values[j, i])) / 2
        )

    with np.errstate(over="raise", invalid="raise"):
        actual = _symmetric_part(values)
    np.testing.assert_array_equal(actual, expected)
