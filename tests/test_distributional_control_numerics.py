"""Finite-output and contract controls for LSS solver boundaries."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

import superglm.distributional.solver.solver as solver_module
from superglm.distributional.family import NaturalLikelihoodEvaluation
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.solver.assembly import (
    _assemble_dense_geometry_from_matrices,
    dense_predictor_matrices,
    validated_dense_penalty,
)
from superglm.distributional.solver.curvature import resolve_curvature
from superglm.distributional.solver.derivatives import (
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.weights import UnsupportedLikelihoodContractError
from superglm.links import IdentityLink

from .test_distributional_chunking import _WrongDerivativeOrderGaussian
from .test_distributional_solver import _ConstantCarrierGaussian, _intercept_layout, _plan

_HUGE = float.fromhex("0x1p+1023")
_MINSUB = float.fromhex("0x0.0000000000001p-1022")
_MAXIMUM = float(np.finfo(np.float64).max)
_UNIT_ROUNDOFF = Fraction(1, 2**53)
_GAMMA_2 = 2 * _UNIT_ROUNDOFF / (1 - 2 * _UNIT_ROUNDOFF)


@pytest.mark.parametrize(
    "diagonal",
    [(_HUGE, _HUGE), (_MINSUB, _MINSUB), (_HUGE, _MINSUB)],
    ids=["large", "subnormal", "mixed"],
)
@pytest.mark.parametrize("route", ["penalty", "curvature", "assembly"])
def test_symmetric_solver_boundaries_preserve_finite_extreme_entries(diagonal, route):
    """The requested average is exactly the supplied diagonal matrix."""
    expected = np.diag(diagonal)
    if route == "penalty":
        actual = validated_dense_penalty(expected, 2)
    elif route == "curvature":
        decision = resolve_curvature("observed", expected)
        assert not decision.retry_required
        assert decision.decomposition is not None
        assert decision.decomposition.rank == 2
        actual = decision.matrix
    else:
        _, layout = _intercept_layout(1)
        geometry = _assemble_dense_geometry_from_matrices(
            layout,
            dense_predictor_matrices(layout),
            np.zeros((1, 2)),
            np.array([[diagonal[0], 0.0, diagonal[1]]]),
            penalty=np.zeros((2, 2)),
            coefficients=np.zeros(2),
        )
        actual = geometry.data_curvature
        np.testing.assert_array_equal(geometry.penalized_curvature, expected)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.isfinite(actual))


class _ScaledIdentityLink(IdentityLink):
    def __init__(self, scale):
        self.scale = scale

    def inverse(self, eta):
        return np.asarray(eta) * self.scale

    def link(self, theta):
        return np.asarray(theta) / self.scale

    def deriv(self, theta):
        return np.full_like(theta, 1.0 / self.scale)

    def deriv_inverse(self, eta):
        return np.full_like(eta, self.scale)


def _observed_cross_product(hessian, left, right):
    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.zeros(1),
        parameter_independent_carrier=np.zeros(1),
        score=np.zeros((1, 2)),
        hessian_packed=np.array([[0.0, hessian, 0.0]]),
    )
    transformed = transform_natural_derivatives(
        evaluation,
        np.zeros((1, 2)),
        (_ScaledIdentityLink(left), _ScaledIdentityLink(right)),
    )
    np.testing.assert_array_equal(transformed.curvature_packed, -transformed.hessian_eta_packed)
    return float(transformed.hessian_eta_packed[0, 1])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize(
    ("hessian", "left", "right"),
    [
        (float.fromhex("0x1p+900"), float.fromhex("0x1p+200"), float.fromhex("0x1p-200")),
        (float.fromhex("0x1p-900"), float.fromhex("0x1p-200"), float.fromhex("0x1p+200")),
        (_MAXIMUM, 2.0, 0.5),
        (_MINSUB, 0.5, 2.0),
        (_MINSUB, float.fromhex("0x1p+1023"), float.fromhex("0x1p-1023")),
        (float.fromhex("0x1p-1022"), float.fromhex("0x1p-52"), 1.0),
        (0.0, _MAXIMUM, 2.0),
    ],
    ids=[
        "intermediate-overflow",
        "intermediate-underflow",
        "maximum-result",
        "subnormal-recovered",
        "subnormal-rescaled",
        "subnormal-result",
        "zero-factor",
    ],
)
def test_observed_link_chain_preserves_exact_dyadic_products(hessian, left, right, sign):
    """Old left-associated products overflow or erase a representable cross term."""
    hessian *= sign
    exact = Fraction.from_float(hessian) * Fraction.from_float(left) * Fraction.from_float(right)
    expected = float(exact)
    assert Fraction.from_float(expected) == exact
    assert _observed_cross_product(hessian, left, right) == expected


@pytest.mark.parametrize("exponent", [-200, 200])
def test_link_chain_handles_ordinary_and_exceptional_rows_together(exponent):
    cross = np.array([_MAXIMUM, 1.0, -_MINSUB, 0.0, -_MAXIMUM, -1.0, _MINSUB])
    packed = np.column_stack((np.zeros(len(cross)), cross, np.zeros(len(cross))))
    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.zeros(len(cross)),
        parameter_independent_carrier=np.zeros(len(cross)),
        score=np.zeros((len(cross), 2)),
        hessian_packed=packed,
    )
    transformed = transform_natural_derivatives(
        evaluation,
        np.zeros((len(cross), 2)),
        (
            _ScaledIdentityLink(np.ldexp(1.0, exponent)),
            _ScaledIdentityLink(np.ldexp(1.0, -exponent)),
        ),
    )
    np.testing.assert_array_equal(transformed.hessian_eta_packed, packed)


@pytest.mark.parametrize(
    ("hessian", "left", "right"),
    [
        (np.finfo(np.float64).tiny, 0.1, float.fromhex("0x1p+1000")),
        (-np.finfo(np.float64).tiny, 0.1, float.fromhex("0x1p+1000")),
        (_MINSUB, 1.5, 1.5),
        (_MINSUB, 0.5, 0.5),
        (0.1, -0.3, 1.7),
    ],
    ids=["amplified-underflow", "signed-underflow", "double-rounding", "below-range", "ordinary"],
)
def test_observed_link_chain_respects_two_multiply_and_subnormal_error_bound(hessian, left, right):
    exact = Fraction.from_float(hessian) * Fraction.from_float(left) * Fraction.from_float(right)
    actual = _observed_cross_product(hessian, left, right)
    # Two normally rounded multiplications, plus one final subnormal rounding.
    allowance = _GAMMA_2 * abs(exact) + Fraction.from_float(_MINSUB) / 2
    assert abs(Fraction.from_float(actual) - exact) <= allowance


def test_fisher_link_chain_preserves_positive_definite_congruence():
    cross = float.fromhex("0x1p-900")
    information = np.array([[float.fromhex("0x1p+400"), cross, float.fromhex("0x1p-400")]])
    actual = transform_natural_information(
        information,
        np.zeros((1, 2)),
        (
            _ScaledIdentityLink(float.fromhex("0x1p-200")),
            _ScaledIdentityLink(float.fromhex("0x1p+200")),
        ),
    )
    # Both natural and predictor matrices are strictly positive definite.
    np.testing.assert_array_equal(actual, [[1.0, cross, 1.0]])


@pytest.mark.parametrize("route", ["observed", "fisher"])
def test_unrepresentable_link_chain_raises_recoverable_numerical_error(route):
    with pytest.raises(_NumericalEvaluationError):
        if route == "observed":
            _observed_cross_product(_HUGE, 2.0, 1.0)
        else:
            transform_natural_information(
                np.array([[_HUGE]]), np.zeros((1, 1)), (_ScaledIdentityLink(2.0),)
            )


def _state_context(family, chunk_size):
    response = np.array([-1.0, 1.0])
    _, layout = _intercept_layout(len(response))
    return solver_module._validated_context(
        family,
        layout,
        response,
        _plan(family, response, np.ones(len(response))),
        np.zeros((2, 2)),
        coefficient_curvature="observed",
        chunk_size=chunk_size,
        coefficient_face=None,
    )


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_wrong_derivative_order_is_hard_before_trial_refusal_in_both_routes(chunk_size):
    context = _state_context(_WrongDerivativeOrderGaussian(), chunk_size)
    with pytest.raises(UnsupportedLikelihoodContractError, match="exact derivative order 0"):
        solver_module._evaluate_state_unmeasured(context, np.zeros(2), derivative_order=0)


@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("failure", ["floating_point", "numerical_evaluation", "invalid_row"])
def test_numeric_value_trial_refusals_remain_recoverable(monkeypatch, chunk_size, failure):
    family = _ConstantCarrierGaussian(0.0)
    context = _state_context(family, chunk_size)
    original_evaluate = family.evaluate_natural

    def evaluate(y, theta, plan, *, derivative_order=2):
        if failure == "floating_point":
            raise FloatingPointError("a numerical trial failure")
        if failure == "numerical_evaluation":
            raise _NumericalEvaluationError("an unrepresentable numerical trial")
        result = original_evaluate(y, theta, plan, derivative_order=derivative_order)
        return replace(result, valid=np.zeros(len(y), dtype=np.bool_))

    monkeypatch.setattr(family, "evaluate_natural", evaluate)
    assert (
        solver_module._evaluate_state_unmeasured(context, np.zeros(2), derivative_order=0) is None
    )
