"""Analytic observed-curvature directions for the two-piece, generalized Pareto and
generalized gamma families.

Each family supplies the derivative of its natural Hessian along a natural
direction and the shared chain rule (``natural_curvature_direction``) maps it
to predictor coordinates.  Where the row log-likelihood needs only ``log`` and
``exp`` (two-piece location form, generalized Pareto) a 160-digit decimal
oracle differentiates the row itself three times; every family is also held
to the Richardson difference of its published curvature at that difference's
resolution.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.distributional.families import (
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    PredictorCurvatureDirectionalFamily,
)
from superglm.distributional.kernels.generalized_gamma import (
    shape_term_d1,
    stirling_remainder_d2,
    stirling_remainder_d3,
)
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    finite_difference_curvature_direction,
)
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

_U = 2.0**-53
_STEP = Decimal("1e-30")


def _plan(family, y, weights=None, semantics="prior"):
    resolved = resolve_likelihood_weights(
        weights, n_observations=len(y), contract=WeightContract(semantics)
    )
    return family.bind_likelihood(y, resolved, COMPLETE_OBSERVATION)


def _links(family):
    return tuple(spec.default_link for spec in family.parameters)


def _severity_rows(seed: int, n_rows: int, k_parameters: int):
    rng = np.random.default_rng(seed)
    y = np.exp(rng.normal(7.0, 1.2, n_rows))
    eta = np.column_stack(
        [np.log(y) + rng.normal(0.0, 0.5, n_rows), rng.normal(0.0, 0.8, n_rows)]
        + [rng.normal(0.0, 1.0, n_rows)] * (k_parameters - 2)
    )
    return y, eta, rng.normal(size=eta.shape)


_CASES = {
    "two-piece log-normal (mean)": lambda: TwoPieceLogNormalLSS(),
    "two-piece log-normal (location)": lambda: TwoPieceLogNormalLSS(parametrisation="location"),
    "two-piece normal": lambda: TwoPieceNormalLSS(),
    "generalized gamma (mean)": lambda: GeneralizedGammaLSS(),
    "generalized gamma (location)": lambda: GeneralizedGammaLSS(parametrisation="location"),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_direction_agrees_with_the_published_curvature(name: str) -> None:
    family = _CASES[name]()
    assert isinstance(family, PredictorCurvatureDirectionalFamily)
    y, eta, direction = _severity_rows(3, 3000, 3)
    if isinstance(family, TwoPieceNormalLSS):
        y = np.log(y)
    if isinstance(family, GeneralizedGammaLSS):
        # Keep sigma |Q| < 1 for negative Q so the mean exists at every stencil point.
        eta[:, 2] = np.random.default_rng(4).uniform(-0.15, 1.2, len(y))
        eta[:, 1] = np.minimum(eta[:, 1], 1.5)
    plan = _plan(family, y)
    links = _links(family)
    analytic = family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    numeric = finite_difference_curvature_direction(
        family, y, eta, direction, links, plan, step=10.0 * DEFAULT_STEP
    )
    scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
    relative = np.abs(numeric.values - analytic) / scale
    # Two-piece rows within a step of the piece boundary see the Hessian's
    # jump; generalized gamma rows with large |Q w| sit at the difference's
    # floor, which is why the median is a decade above the other families'.
    assert np.quantile(relative, 0.99) < 1.0e-7
    assert np.median(relative) < 1.0e-9


def test_generalized_pareto_direction_agrees_with_the_published_curvature() -> None:
    family = GeneralizedParetoLSS()
    rng = np.random.default_rng(6)
    n_rows = 3000
    y = np.exp(rng.normal(7.0, 2.0, n_rows))
    eta = np.column_stack((rng.normal(7.0, 1.0, n_rows), rng.normal(0.0, 2.0, n_rows)))
    direction = rng.normal(size=eta.shape)
    plan = _plan(family, y)
    links = _links(family)
    analytic = family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    numeric = finite_difference_curvature_direction(
        family, y, eta, direction, links, plan, step=10.0 * DEFAULT_STEP
    )
    scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
    relative = np.abs(numeric.values - analytic) / scale
    assert np.max(relative) < 1.0e-8
    assert np.median(relative) < 1.0e-10


def _decimal_direction(log_likelihood, eta, direction):
    """``-d/dt Hessian`` of ``log_likelihood`` along ``direction`` at 160 digits."""
    with localcontext() as context:
        context.prec = 160
        k_parameters = len(eta)
        point = [Decimal(float(value)) for value in eta]
        out = []
        for left in range(k_parameters):
            for right in range(left, k_parameters):
                total = Decimal(0)
                for axis in range(k_parameters):
                    stencil = Decimal(0)
                    for signs in np.ndindex(2, 2, 2):
                        shifted = list(point)
                        sign = 1
                        for index, bit in zip((left, right, axis), signs, strict=True):
                            step = _STEP if bit == 0 else -_STEP
                            sign *= 1 if bit == 0 else -1
                            shifted[index] += step
                        stencil += sign * log_likelihood(shifted)
                    total += stencil / (8 * _STEP**3) * Decimal(float(direction[axis]))
                out.append(-total)
        return out


def _assert_natural_oracle(
    direction_fn, log_likelihood, theta, units: int, term_scale=None
) -> None:
    """``dH(v)`` in natural coordinates against the decimal oracle, ``v`` each unit axis.

    The predictor chain rule is shared and checked through the NB2 and
    log-normal oracles; this isolates the family's own third derivatives.
    Each entry is a short sum of products of singly rounded factors, so its
    error is a small multiple of ``u`` times the row's third-derivative scale
    ``max |T|`` (entries that cancel, such as ``v - a`` at ``t = 1`` in the
    generalized Pareto, are bounded by that scale, not by their own size).
    ``term_scale`` adds a per-row term scale where one entry's terms can exceed it.
    """
    n_rows, k_parameters = theta.shape
    analytic = []
    for axis in range(k_parameters):
        natural = np.zeros_like(theta)
        natural[:, axis] = 1.0
        analytic.append(direction_fn(natural))
    for row in range(n_rows):
        oracle = []
        for axis in range(k_parameters):
            natural = np.zeros(k_parameters)
            natural[axis] = 1.0
            oracle.append(
                _decimal_direction(
                    lambda point, row=row: log_likelihood(row, point), theta[row], natural
                )
            )
        scale = max(abs(value) for values in oracle for value in values)
        if term_scale is not None:
            scale = max(scale, Decimal(float(term_scale[row])))
        for axis in range(k_parameters):
            for channel, value in enumerate(oracle[axis]):
                error = abs(Decimal(float(-analytic[axis][row, channel])) - value)
                assert error <= Decimal(units * _U) * scale, (row, axis, channel)


def test_two_piece_hessian_direction_matches_the_decimal_oracle() -> None:
    from superglm.distributional.kernels.two_piece import location_hessian_direction

    rng = np.random.default_rng(8)
    n_rows = 20
    t = rng.normal(0.0, 3.0, n_rows)
    theta = np.column_stack(
        (
            t + rng.normal(0.0, 2.0, n_rows),
            np.exp(rng.normal(0.0, 1.0, n_rows)),
            rng.uniform(-0.9, 0.9, n_rows),
        )
    )

    def log_likelihood(row, point):
        z = (Decimal(float(t[row])) - point[0]) / point[1]
        s = 1 - point[2] if z < 0 else 1 + point[2]
        return -point[1].ln() - (z / s) ** 2 / 2

    _assert_natural_oracle(
        lambda v: location_hessian_direction(
            t, theta[:, 0], theta[:, 1], theta[:, 2], v, np.ones(n_rows)
        ),
        log_likelihood,
        theta,
        32,
    )


def test_generalized_pareto_hessian_direction_matches_the_decimal_oracle() -> None:
    from superglm.distributional.kernels.generalized_pareto import scale_hessian_direction

    rng = np.random.default_rng(9)
    n_rows = 20
    y = np.exp(rng.normal(7.0, 2.0, n_rows))
    # Shapes from 1e-3 to near one: the series region (z < 1/2), the tail
    # numerator's series (z/(1+z) < 0.8) and its direct form.
    theta = np.column_stack(
        (np.exp(rng.normal(7.0, 1.0, n_rows)), np.exp(rng.uniform(-7.0, -0.01, n_rows)))
    )

    def log_likelihood(row, point):
        response = Decimal(float(y[row]))
        return -point[0].ln() - (1 + 1 / point[1]) * (1 + point[1] * response / point[0]).ln()

    # T_xixixi = t^4 F(z) - 2 v^3 (or its tail form) crosses zero near t = 4/3,
    # between terms of size v^4: its rounding is relative to those terms.
    t = y / theta[:, 0]
    z = theta[:, 1] * t
    v = t / (1.0 + z)
    u = z / (1.0 + z)
    tail = (6.0 * np.log1p(z) + 6.0 * u + 3.0 * u * u + 2.0 * u**3) / theta[:, 1] ** 4
    term_scale = np.where(z < 0.5, 1.5 * t**4, tail) + 2.0 * v**3
    _assert_natural_oracle(
        lambda v: scale_hessian_direction(y, theta[:, 0], theta[:, 1], v, np.ones(n_rows)),
        log_likelihood,
        theta,
        64,
        term_scale,
    )


def test_shape_term_derivative_is_continuous_across_its_series_switch() -> None:
    """The series and direct forms of ``f'(Q)`` agree where they meet (``k = 30``)."""
    switch = 1.0 / np.sqrt(30.0)
    shapes = switch * np.array([1.0 - 1e-9, 1.0 + 1e-9])
    values = shape_term_d1(shapes)
    np.testing.assert_allclose(values[0], values[1], rtol=1e-8)
    # Direct form against the Stirling remainders, sign-odd in Q.
    q = np.array([0.4, -0.4, 1.3])
    k = 1.0 / (q * q)
    expected = (
        2.0
        * (
            18.0 * k * k * stirling_remainder_d2(k)
            + 4.0 * k**3 * stirling_remainder_d3(k)
            + 12.0
            * k
            * (
                np.vectorize(lambda x: __import__("scipy").special.digamma(x))(k)
                - np.log(k)
                + 0.5 / k
            )
        )
        / q**3
    )
    np.testing.assert_allclose(shape_term_d1(q), expected, rtol=1e-11)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_tweedie_direction_agrees_with_the_published_curvature(semantics: str) -> None:
    """Zero rows (closed form) and positive rows (series cumulants) across the power range."""
    from superglm.distributional.families import TweedieLSS

    family = TweedieLSS()
    rng = np.random.default_rng(10)
    n_rows = 2000
    exposure = rng.uniform(0.1, 1.0, n_rows)
    claims = rng.poisson(0.3 * exposure)
    y = np.where(claims > 0, rng.gamma(2.0, 800.0, n_rows) * claims / exposure, 0.0)
    weights = exposure if semantics == "prior" else rng.integers(1, 4, n_rows).astype(np.float64)
    eta = np.column_stack(
        (
            np.log(250.0) + rng.normal(0.0, 0.4, n_rows),
            rng.normal(6.0, 0.8, n_rows),
            rng.uniform(-2.5, 2.5, n_rows),
        )
    )
    direction = rng.normal(size=eta.shape)
    plan = _plan(family, y, weights, semantics)
    links = _links(family)
    analytic = family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    numeric = finite_difference_curvature_direction(
        family, y, eta, direction, links, plan, step=10.0 * DEFAULT_STEP
    )
    scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
    relative = np.abs(numeric.values - analytic) / scale
    # Near the lower power wall the Richardson difference itself is off by up
    # to 4e-6 of the row scale; at those rows 40-digit mpmath puts the
    # analytic direction within 1e-13 and the difference at its 4e-6.
    assert np.quantile(relative, 0.99) < 1.0e-7
    assert np.max(relative) < 1.0e-5
    assert np.median(relative) < 1.0e-10


def test_bounded_power_link_third_derivative() -> None:
    from superglm.distributional.families.tweedie import BoundedPowerLink

    link = BoundedPowerLink()
    eta = np.linspace(-30.0, 30.0, 121)
    with localcontext() as context:
        context.prec = 60
        for value, computed in zip(eta, link.deriv3_inverse(eta), strict=True):
            e = Decimal(float(value))
            p = 1 / (1 + (-e).exp())
            q = 1 - p
            exact = Decimal(link.upper - link.lower) * p * q * (1 - 6 * p * q)
            assert abs(Decimal(float(computed)) - exact) <= Decimal(16 * _U) * abs(exact) * 7
