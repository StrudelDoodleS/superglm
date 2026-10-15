"""Analytic NB2 third derivatives for the LAML endgame.

Held to an independent 120-digit oracle within the bound the kernel
certifies, and to the Richardson difference of the published curvature at
that difference's resolution.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.distributional.families import NegativeBinomialLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    finite_difference_curvature_direction,
)
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink, LogLink
from tests._distributional_family_kernels import negative_binomial as nb_kernel

_U = 2.0**-53


def _oracle_third(count: int, mean: float, theta: float) -> list[Decimal]:
    """``l_aaa, l_aab, l_abb, l_bbb`` at 120 digits from the theta derivatives of the row."""
    with localcontext() as context:
        context.prec = 120
        y, mu, th = Decimal(count), Decimal(mean), Decimal(theta)
        t = mu + th
        d1 = sum((1 / (th + j) for j in range(count)), Decimal(0))
        d2 = -sum((1 / (th + j) ** 2 for j in range(count)), Decimal(0))
        d3 = 2 * sum((1 / (th + j) ** 3 for j in range(count)), Decimal(0))
        f1 = d1 + (th / t).ln() + (mu - y) / t
        f2 = d2 + 1 / th - 2 / t + (y + th) / t**2
        f3 = d3 - 1 / th**2 + 3 / t**2 - 2 * (y + th) / t**3
        q, p, r = mu / t, th / t, (th - mu) / t
        s = q * p
        return [
            -(y + th) * r * s,
            s * (y * r - 2 * mu * p),
            -(y - mu) * r * s,
            th * f1 + 3 * th**2 * f2 + th**3 * f3,
            s * (abs(y * r) + 2 * mu * p),
        ]


def _oracle_rows(seed: int, n_rows: int):
    rng = np.random.default_rng(seed)
    theta = np.exp(rng.uniform(-8.0, 40.0, n_rows))
    mean = theta * np.exp(rng.uniform(np.log(2.0**-52), np.log(2.0**26), n_rows))
    counts = np.minimum(rng.poisson(np.minimum(mean, 300.0)), 400).astype(np.float64)
    return counts, mean, theta


def test_theta_third_derivative_is_inside_its_certified_bound() -> None:
    """Near Poisson ``l_bbb`` cancels by ``theta^2``; the chosen route still certifies its error."""
    counts, mean, theta = _oracle_rows(7, 600)
    counts[:3], mean[:3], theta[:3] = [3.0, 0.0, 2.0], [0.5, 1e-3, 2.0**-30], [1e15, 1e12, 2.0**20]
    third, error = nb_kernel.negative_binomial_log_third_derivatives(
        counts, mean, theta, np.ones(len(counts)), "prior"
    )
    gamma_11 = 11 * _U / (1 - 11 * _U)
    for row in range(len(counts)):
        oracle = _oracle_third(int(counts[row]), float(mean[row]), float(theta[row]))
        assert abs(Decimal(float(third[row, 3])) - oracle[3]) <= Decimal(float(error[row]))
        scales = (abs(oracle[0]), oracle[4], abs(oracle[2]))
        for channel, scale in enumerate(scales):
            difference = abs(Decimal(float(third[row, channel])) - oracle[channel])
            assert difference <= Decimal(gamma_11) * scale


def test_theta_third_derivative_bound_is_not_vacuous() -> None:
    counts, mean, theta = _oracle_rows(8, 400)
    third, error = nb_kernel.negative_binomial_log_third_derivatives(
        counts, mean, theta, np.ones(len(counts)), "prior"
    )
    resolvable = np.abs(third[:, 3]) > 0.0
    assert np.median(error[resolvable] / np.abs(third[resolvable, 3])) < 1e-12


def _plan(response, weights, semantics):
    resolved = resolve_likelihood_weights(
        weights, n_observations=len(response), contract=WeightContract(semantics)
    )
    return NegativeBinomialLS().bind_likelihood(response, resolved, COMPLETE_OBSERVATION)


@pytest.mark.parametrize("semantics", ["prior", "frequency", "exposure"])
def test_analytic_direction_agrees_with_the_published_curvature(semantics: str) -> None:
    """Consistency with the Richardson difference of the published curvature.

    The difference is a truncation estimate whose certificate is not a bound
    (it can miss by a few times on single rows), so this checks agreement at
    the difference's resolution; the accuracy claim is the oracle test above.
    """
    rng = np.random.default_rng(4)
    n_rows = 3000
    counts = rng.poisson(0.3, n_rows).astype(np.float64)
    weights = None
    response = counts
    if semantics == "frequency":
        weights = rng.integers(1, 4, n_rows).astype(np.float64)
    elif semantics == "exposure":
        weights = rng.uniform(0.1, 1.0, n_rows)
        response = counts / weights
    plan = _plan(response, weights, "frequency" if semantics == "frequency" else "prior")
    eta = np.column_stack(
        (np.log(0.3) + rng.normal(0.0, 0.5, n_rows), rng.normal(0.0, 3.0, n_rows))
    )
    family, links = NegativeBinomialLS(), (LogLink(), LogLink())
    for direction in (np.tile([1.0, 0.0], (n_rows, 1)), rng.normal(size=eta.shape)):
        analytic = family.predictor_curvature_directional_derivative(
            response, eta, direction, links, plan
        )
        numeric = finite_difference_curvature_direction(
            family, response, eta, direction, links, plan, step=10.0 * DEFAULT_STEP
        )
        scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
        relative = np.abs(numeric.values - analytic) / scale
        assert np.max(relative) < 1.0e-8
        assert np.median(relative) < 1.0e-10


def test_laml_derivatives_take_the_analytic_direction() -> None:
    """The endgame differentiates NB2 curvature analytically instead of by stencil."""
    from superglm.distributional.family import PredictorCurvatureDirectionalFamily
    from superglm.distributional.smoothing.derivatives import _analytic_first_derivative

    response = np.array([0.0, 1.0, 2.0, 0.0, 5.0])
    plan = _plan(response, None, "prior")
    eta = np.column_stack((np.full(5, 0.1), np.full(5, 1.5)))
    family = NegativeBinomialLS()
    assert isinstance(family, PredictorCurvatureDirectionalFamily)
    derivative = _analytic_first_derivative(
        family, response, eta, (LogLink(), LogLink()), plan, 5, 2
    )
    assert derivative is not None


def test_analytic_direction_refuses_links_it_does_not_differentiate() -> None:
    response = np.array([0.0, 1.0, 2.0])
    plan = _plan(response, None, "prior")
    eta = np.zeros((3, 2))
    with pytest.raises(UnsupportedLikelihoodContractError):
        NegativeBinomialLS().predictor_curvature_directional_derivative(
            response, eta, eta, (IdentityLink(), LogLink()), plan
        )
