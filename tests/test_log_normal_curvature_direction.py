"""Analytic observed-curvature directions for LogNormalLS and the Gaussian sibling it reuses.

The oracle differentiates the row log-likelihood itself three times by
central differences in 160-digit decimal arithmetic (step 1e-30, so the
truncation is ~1e-60 and the rounding ~1e-70), independent of every formula
under test.  The analytic channels are sums of a few products of singly
rounded factors, so they must agree to a small multiple of the unit roundoff
of the row's largest channel.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.distributional.families import GaussianLS, LogNormalLS
from superglm.distributional.families.gaussian import LowerBoundedLogLink
from superglm.distributional.family import COMPLETE_OBSERVATION, PredictorCurvatureDirectionalFamily
from superglm.distributional.kernels.gaussian import gaussian_predictor_curvature_directional
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

_U = 2.0**-53
_STEP = Decimal("1e-30")


def _row_log_likelihood(first: Decimal, second: Decimal, z: Decimal, floor: Decimal, mean_form):
    sigma = floor + second.exp()
    location = first - sigma * sigma / 2 if mean_form else first
    residual = z - location
    return -sigma.ln() - residual * residual / (2 * sigma * sigma)


def _oracle_direction(eta, direction, z, floor, mean_form):
    with localcontext() as context:
        context.prec = 160
        point = [Decimal(float(eta[0])), Decimal(float(eta[1]))]
        zd, fd = Decimal(float(z)), Decimal(float(floor))
        tensor = {}
        for i in range(2):
            for j in range(i, 2):
                for k in range(2):
                    total = Decimal(0)
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                shifted = list(point)
                                shifted[i] += si * _STEP
                                shifted[j] += sj * _STEP
                                shifted[k] += sk * _STEP
                                total += (
                                    si
                                    * sj
                                    * sk
                                    * _row_log_likelihood(shifted[0], shifted[1], zd, fd, mean_form)
                                )
                    tensor[i, j, k] = total / (8 * _STEP**3)
        d = [Decimal(float(direction[0])), Decimal(float(direction[1]))]
        return [
            -(tensor[0, 0, 0] * d[0] + tensor[0, 0, 1] * d[1]),
            -(tensor[0, 1, 0] * d[0] + tensor[0, 1, 1] * d[1]),
            -(tensor[1, 1, 0] * d[0] + tensor[1, 1, 1] * d[1]),
        ]


def _rows(seed: int, n_rows: int):
    rng = np.random.default_rng(seed)
    y = np.exp(rng.normal(7.0, 1.2, n_rows))
    eta = np.column_stack((np.log(y) + rng.normal(0.0, 0.5, n_rows), rng.normal(0.0, 2.0, n_rows)))
    # Both ends of the scale link: sigma at its floor and sigma far above it.
    eta[:10, 1] = rng.uniform(-9.0, -5.0, 10)
    eta[10:20, 1] = rng.uniform(4.0, 7.0, 10)
    return y, eta, rng.normal(size=eta.shape)


def _plan(family, y):
    weights = resolve_likelihood_weights(
        None, n_observations=len(y), contract=WeightContract("prior")
    )
    return family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)


def _assert_near_oracle(analytic, eta, direction, z, floor, mean_form, units):
    for row in range(len(z)):
        oracle = _oracle_direction(eta[row], direction[row], z[row], floor, mean_form)
        scale = max(abs(value) for value in oracle)
        for channel in range(3):
            error = abs(Decimal(float(analytic[row, channel])) - oracle[channel])
            assert error <= Decimal(units * _U) * scale, (row, channel)


@pytest.mark.parametrize("parametrisation", ["mean", "location"])
def test_log_normal_direction_matches_the_decimal_oracle(parametrisation: str) -> None:
    family = LogNormalLS(parametrisation=parametrisation)
    y, eta, direction = _rows(3, 60)
    links = (LogLink() if parametrisation == "mean" else IdentityLink(), LowerBoundedLogLink(0.01))
    analytic = np.asarray(
        family.predictor_curvature_directional_derivative(
            y, eta, direction, links, _plan(family, y)
        )
    )
    _assert_near_oracle(analytic, eta, direction, np.log(y), 0.01, parametrisation == "mean", 32)


def test_gaussian_direction_keeps_accuracy_far_above_the_scale_floor() -> None:
    """``1 - u`` with ``u = (sigma - floor)/sigma`` is formed as ``floor/sigma``.

    Formed as ``1 - u`` it cancels as sigma leaves its floor: at ``sigma = 1e3``
    and floor ``0.01`` the scale-scale channel lost ``1e-12`` of the row scale.
    """
    rng = np.random.default_rng(5)
    n_rows = 40
    eta = np.column_stack((rng.normal(0.0, 1.0, n_rows), rng.uniform(4.0, 8.0, n_rows)))
    z = eta[:, 0] + rng.normal(0.0, 0.01, n_rows) * np.exp(eta[:, 1])
    direction = rng.normal(size=eta.shape)
    analytic = gaussian_predictor_curvature_directional(
        z, eta, direction, np.ones(n_rows), "prior", scale_floor=0.01
    )
    _assert_near_oracle(analytic, eta, direction, z, 0.01, False, 32)


@pytest.mark.parametrize("parametrisation", ["mean", "location"])
def test_log_normal_direction_agrees_with_the_published_curvature(parametrisation: str) -> None:
    family = LogNormalLS(parametrisation=parametrisation)
    y, eta, direction = _rows(4, 4000)
    plan = _plan(family, y)
    links = (LogLink() if parametrisation == "mean" else IdentityLink(), LowerBoundedLogLink(0.01))
    analytic = family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    numeric = finite_difference_curvature_direction(
        family, y, eta, direction, links, plan, step=10.0 * DEFAULT_STEP
    )
    scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
    relative = np.abs(numeric.values - analytic) / scale
    assert np.max(relative) < 1.0e-8
    assert np.median(relative) < 1.0e-10


def test_log_normal_direction_is_offered_and_refuses_other_links() -> None:
    family = LogNormalLS()
    assert isinstance(family, PredictorCurvatureDirectionalFamily)
    y, eta, direction = _rows(6, 30)
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.predictor_curvature_directional_derivative(
            y, eta, direction, (IdentityLink(), LowerBoundedLogLink(0.01)), _plan(family, y)
        )
    assert isinstance(GaussianLS(), PredictorCurvatureDirectionalFamily)
