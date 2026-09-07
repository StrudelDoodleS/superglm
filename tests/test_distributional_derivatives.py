from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from superglm.distributional.derivatives import transform_natural_derivatives
from superglm.distributional.family import NaturalLikelihoodEvaluation
from superglm.distributional.packing import packed_pairs
from superglm.links import IdentityLink, InverseLink, LogitLink, LogLink


@dataclass(frozen=True)
class _LowerBoundedLogLink:
    floor: float

    def link(self, mu: np.ndarray) -> np.ndarray:
        return np.log(mu - self.floor)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return self.floor + np.exp(eta)

    def deriv(self, mu: np.ndarray) -> np.ndarray:
        return 1.0 / (mu - self.floor)

    def deriv_inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)

    def deriv2_inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)


def _quadratic_fixture():
    links = (
        IdentityLink(),
        LogLink(),
        LogitLink(),
        InverseLink(),
        _LowerBoundedLogLink(0.01),
    )
    eta = np.array(
        [
            [0.2, -0.3, -0.7, 1.4, -1.0],
            [-0.4, 0.1, 0.6, 2.0, -0.2],
            [0.8, 0.4, -0.1, 1.1, 0.3],
        ],
        dtype=float,
    )
    weights = np.array([0.4, 1.7, 2.3])
    linear = np.array(
        [
            [0.7, -0.1, 0.3, 0.5, -0.6],
            [-0.2, 0.4, 0.8, -0.3, 0.2],
            [0.1, -0.5, 0.2, 0.9, 0.4],
        ]
    )
    q = np.array(
        [
            [1.3, 0.12, -0.08, 0.04, 0.06],
            [0.12, 0.9, 0.10, -0.03, 0.07],
            [-0.08, 0.10, 1.1, 0.09, -0.05],
            [0.04, -0.03, 0.09, 0.8, 0.11],
            [0.06, 0.07, -0.05, 0.11, 1.5],
        ]
    )

    def objective(row: int, eta_row: np.ndarray) -> float:
        theta = np.array([link.inverse(eta_row[k : k + 1])[0] for k, link in enumerate(links)])
        return float(weights[row] * (linear[row] @ theta - 0.5 * theta @ q @ theta))

    theta = np.column_stack([link.inverse(eta[:, k]) for k, link in enumerate(links)])
    score = weights[:, None] * (linear - theta @ q)
    hessian = np.empty((len(eta), len(packed_pairs(len(links)))))
    for packed, (left, right) in enumerate(packed_pairs(len(links))):
        hessian[:, packed] = -weights * q[left, right]
    log_likelihood = np.array([objective(row, eta[row]) for row in range(len(eta))])
    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=log_likelihood,
        parameter_independent_carrier=np.array([0.1, -0.2, 0.3]),
        score=score,
        hessian_packed=hessian,
        valid=np.array([True, False, True]),
    )
    return links, eta, weights, objective, evaluation


def _numerical_row_derivatives(objective, row: int, point: np.ndarray):
    step = 1.0e-4
    n_parameters = len(point)
    gradient = np.empty(n_parameters)
    hessian = np.empty((n_parameters, n_parameters))
    center = objective(row, point)
    for left in range(n_parameters):
        direction_left = np.zeros(n_parameters)
        direction_left[left] = step
        plus = objective(row, point + direction_left)
        minus = objective(row, point - direction_left)
        gradient[left] = (plus - minus) / (2.0 * step)
        hessian[left, left] = (plus - 2.0 * center + minus) / step**2
        for right in range(left + 1, n_parameters):
            direction_right = np.zeros(n_parameters)
            direction_right[right] = step
            cross = (
                objective(row, point + direction_left + direction_right)
                - objective(row, point + direction_left - direction_right)
                - objective(row, point - direction_left + direction_right)
                + objective(row, point - direction_left - direction_right)
            ) / (4.0 * step**2)
            hessian[left, right] = cross
            hessian[right, left] = cross
    return gradient, hessian


def test_chain_rule_matches_independent_numerical_oracle_for_all_link_types() -> None:
    links, eta, _weights, objective, evaluation = _quadratic_fixture()
    result = transform_natural_derivatives(evaluation, eta, links)

    numerical_score = np.empty_like(result.score_eta)
    numerical_hessian = np.empty((len(eta), len(links), len(links)))
    for row in range(len(eta)):
        numerical_score[row], numerical_hessian[row] = _numerical_row_derivatives(
            objective, row, eta[row]
        )

    k = len(links)
    curvature_packed = result.curvature_packed
    curvature = np.empty((*curvature_packed.shape[:-1], k, k), dtype=curvature_packed.dtype)
    for channel, (left, right) in enumerate(packed_pairs(k)):
        curvature[..., left, right] = curvature_packed[..., channel]
        curvature[..., right, left] = curvature_packed[..., channel]
    hessian_eta_packed = result.hessian_eta_packed
    hessian_eta = np.empty((*hessian_eta_packed.shape[:-1], k, k), dtype=hessian_eta_packed.dtype)
    for channel, (left, right) in enumerate(packed_pairs(k)):
        hessian_eta[..., left, right] = hessian_eta_packed[..., channel]
        hessian_eta[..., right, left] = hessian_eta_packed[..., channel]

    np.testing.assert_allclose(result.score_eta, numerical_score, rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(
        curvature,
        -numerical_hessian,
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        hessian_eta,
        numerical_hessian,
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_array_equal(result.valid, evaluation.valid)
    np.testing.assert_array_equal(
        result.optimizing_log_likelihood,
        evaluation.optimizing_log_likelihood,
    )
    np.testing.assert_array_equal(
        result.parameter_independent_carrier,
        evaluation.parameter_independent_carrier,
    )
    np.testing.assert_array_equal(
        result.reported_log_likelihood,
        evaluation.reported_log_likelihood,
    )
    assert result.valid is not None and not result.valid.flags.writeable
    assert not result.score_eta.flags.writeable
    assert not result.curvature_packed.flags.writeable


def test_oracle_detects_double_weighting_and_double_negation_mutations() -> None:
    links, eta, weights, objective, evaluation = _quadratic_fixture()
    result = transform_natural_derivatives(evaluation, eta, links)
    numerical_score = np.empty_like(result.score_eta)
    numerical_hessian = np.empty((len(eta), len(links), len(links)))
    for row in range(len(eta)):
        numerical_score[row], numerical_hessian[row] = _numerical_row_derivatives(
            objective, row, eta[row]
        )

    double_weighted = result.score_eta * weights[:, None]
    k = len(links)
    curvature_packed = result.curvature_packed
    curvature = np.empty((*curvature_packed.shape[:-1], k, k), dtype=curvature_packed.dtype)
    for channel, (left, right) in enumerate(packed_pairs(k)):
        curvature[..., left, right] = curvature_packed[..., channel]
        curvature[..., right, left] = curvature_packed[..., channel]
    double_negated = -curvature
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(double_weighted, numerical_score, rtol=3e-7, atol=3e-8)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(double_negated, -numerical_hessian, rtol=3e-6, atol=3e-6)


def test_diagonal_chain_term_uses_natural_score_and_second_inverse_derivative() -> None:
    eta = np.array([[-0.5], [0.25]])
    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.array([0.0, 0.0]),
        parameter_independent_carrier=np.array([2.0, 3.0]),
        score=np.array([[2.0], [-3.0]]),
        hessian_packed=np.array([[-0.7], [-0.7]]),
    )
    link = _LowerBoundedLogLink(0.01)

    result = transform_natural_derivatives(evaluation, eta, (link,))
    first = np.exp(eta[:, 0])
    expected_hessian = -0.7 * first**2 + evaluation.score[:, 0] * first

    np.testing.assert_allclose(result.hessian_eta_packed[:, 0], expected_hessian)
    np.testing.assert_allclose(result.curvature_packed[:, 0], -expected_hessian)


class _MissingSecondDerivativeLink:
    def link(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def deriv(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    def deriv_inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.ones_like(eta)


class _NonfiniteDerivativeLink(_MissingSecondDerivativeLink):
    def deriv2_inverse(self, eta: np.ndarray) -> np.ndarray:
        result = np.ones_like(eta)
        result[0] = np.nan
        return result


class _ExplodingDerivativeLink(IdentityLink):
    def deriv_inverse(self, eta: np.ndarray) -> np.ndarray:
        raise AssertionError("lower-order evaluation reached the link derivative")

    def deriv2_inverse(self, eta: np.ndarray) -> np.ndarray:
        raise AssertionError("lower-order evaluation reached the second link derivative")


@pytest.mark.parametrize("derivative_order", [0, 1])
def test_transform_refuses_incomplete_natural_derivatives_before_link_work(
    derivative_order: int,
) -> None:
    """Kills optional-array dereferences and link work before the order gate."""

    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.zeros(2),
        parameter_independent_carrier=np.zeros(2),
        score=np.zeros((2, 1)) if derivative_order == 1 else None,
        hessian_packed=None,
    )

    with pytest.raises(ValueError, match="order 2"):
        transform_natural_derivatives(
            evaluation,
            np.zeros((2, 1)),
            (_ExplodingDerivativeLink(),),
        )


def test_transform_rejects_shape_link_and_nonfinite_derivative_contract_violations() -> None:
    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.zeros(2),
        parameter_independent_carrier=np.zeros(2),
        score=np.zeros((2, 1)),
        hessian_packed=np.zeros((2, 1)),
    )

    with pytest.raises(ValueError, match="eta.*shape"):
        transform_natural_derivatives(evaluation, np.zeros((2, 2)), (IdentityLink(),))
    with pytest.raises(ValueError, match="one link per parameter"):
        transform_natural_derivatives(evaluation, np.zeros((2, 1)), ())
    with pytest.raises(NotImplementedError, match="deriv2_inverse"):
        transform_natural_derivatives(
            evaluation, np.zeros((2, 1)), (_MissingSecondDerivativeLink(),)
        )
    with pytest.raises(ValueError, match="finite.*deriv2_inverse"):
        transform_natural_derivatives(evaluation, np.zeros((2, 1)), (_NonfiniteDerivativeLink(),))
    bad_eta = np.zeros((2, 1))
    bad_eta[0, 0] = np.inf
    with pytest.raises(ValueError, match="eta.*finite"):
        transform_natural_derivatives(evaluation, bad_eta, (IdentityLink(),))
