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


def test_generalized_gamma_refuses_a_valid_row_whose_direction_overflows() -> None:
    """Only rows the family flags invalid publish zero; a valid overflow is refused.

    At ``mu = -1e62``, ``sigma = 1``, ``Q = 1e-62`` the row is valid (``u = Q w``
    is about 1, so ``e^u`` is finite) and its Hessian is finite (``w^4`` terms
    near ``1e248``), but ``T_QQQ`` carries ``w^5 T4(u)``, which overflows.  At
    7305fe25 the kernel zeroed that row and the chain rule still added its
    ``H``, ``g''`` and ``g'''`` terms, a silently wrong derivative; now the
    non-finite row reaches the refusal in ``natural_curvature_direction``.
    A row whose ``e^u`` overflows is invalid and still publishes zero.
    """
    from superglm.distributional.kernels.generalized_gamma import location_hessian_direction

    y = np.array([2.0, 2.0, 2.0])
    mu = np.array([0.5, -1.0e62, 0.5])
    sigma = np.ones(3)
    q = np.array([0.3, 1.0e-62, 5000.0])
    with np.errstate(over="ignore", invalid="ignore"):
        rows = location_hessian_direction(y, mu, sigma, q, np.ones((3, 3)), np.ones(3))
    assert np.all(np.isfinite(rows[0]))
    assert not np.all(np.isfinite(rows[1]))
    assert np.all(rows[2] == 0.0)

    family = GeneralizedGammaLSS(parametrisation="location")
    plan = _plan(family, y[:2])
    links = _links(family)
    eta = np.column_stack((mu[:2], links[1].link(sigma[:2]), q[:2]))
    evaluation = family.evaluate_natural(
        y[:2], np.column_stack((mu[:2], sigma[:2], q[:2])), plan, derivative_order=2
    )
    assert np.all(evaluation.valid)
    assert np.all(np.isfinite(evaluation.hessian_packed))
    with pytest.raises(ValueError, match="not representable"):
        family.predictor_curvature_directional_derivative(y[:2], eta, np.ones((2, 3)), links, plan)


def test_generalized_gamma_mean_form_keeps_a_valid_overflow_non_finite(monkeypatch) -> None:
    """The mean form zeroes only rows whose mean does not exist or whose base is invalid.

    No float64 mean-form row is known to overflow while valid, so the loading's
    third derivative is forced to overflow on one valid row: that row must stay
    non-finite rather than read as zero curvature change.
    """
    import superglm.distributional.kernels.generalized_gamma as kernel

    original = kernel._log_mean_loading_third

    def overflowing(sigma, shape):
        values = list(original(sigma, shape))
        values[-1] = np.array(values[-1], dtype=np.float64, copy=True)
        values[-1][1] = np.inf
        return tuple(values)

    monkeypatch.setattr(kernel, "_log_mean_loading_third", overflowing)
    y = np.array([2.0, 3.0])
    with np.errstate(over="ignore", invalid="ignore"):
        rows = kernel.mean_hessian_direction(
            y,
            np.array([2.5, 2.5]),
            np.array([0.4, 0.4]),
            np.array([0.6, 0.6]),
            np.ones((2, 3)),
            np.ones(2),
        )
    assert np.all(np.isfinite(rows[0]))
    assert not np.all(np.isfinite(rows[1]))


def test_two_piece_refuses_the_analytic_direction_across_its_join() -> None:
    """At ``t == mu`` with nonzero skew the direction is refused when it moves ``mu``.

    The Hessian jumps there (``s`` is ``1 + eps`` on the right, ``1 - eps`` on
    the left), so a direction that moves the location has only one-sided
    derivatives; at 7305fe25 the analytic path returned the right piece's as
    exact.  The refusal sends the caller to the Richardson difference, which
    v0.40.0 used for every two-piece row and whose certificate records the
    jump.  Directions that leave ``mu`` fixed keep ``z = 0`` on the right piece
    and are still analytic, as is a zero-skew row, where nothing jumps.
    """
    from superglm.distributional.kernels.two_piece import log_mean_loading
    from superglm.distributional.smoothing.derivatives import _analytic_first_derivative
    from superglm.distributional.smoothing.endpoint_direction import (
        analytic_curvature_direction,
    )
    from superglm.distributional.weights import UnsupportedLikelihoodContractError

    y = np.array([1.0, 1.3, 0.7])
    family = TwoPieceNormalLSS()
    links = _links(family)
    plan = _plan(family, y)
    eta = np.column_stack(
        (np.ones(3), links[1].link(np.full(3, 0.8)), links[2].link(np.full(3, 0.3)))
    )
    moving = np.tile([0.3, -0.5, 0.8], (3, 1))
    with pytest.raises(UnsupportedLikelihoodContractError, match="join at row 0"):
        family.predictor_curvature_directional_derivative(y, eta, moving, links, plan)
    assert analytic_curvature_direction(family, y, eta, moving, links, plan) is None
    assert _analytic_first_derivative(family, y, eta, links, plan, 3, 3) is None

    fixed = np.tile([0.0, 1.0, -0.4], (3, 1))
    analytic = family.predictor_curvature_directional_derivative(y, eta, fixed, links, plan)
    numeric = finite_difference_curvature_direction(family, y, eta, fixed, links, plan)
    scale = np.max(np.abs(analytic), axis=1, keepdims=True)
    assert np.all(np.abs(numeric.values - analytic) <= numeric.certificate + 1e-9 * scale)

    symmetric = eta.copy()
    symmetric[:, 2] = links[2].link(np.zeros(3))
    family.predictor_curvature_directional_derivative(y, symmetric, moving, links, plan)

    mean_family = TwoPieceLogNormalLSS(parametrisation="mean")
    mean_links = _links(mean_family)
    response = np.array([2.0, 3.0])
    sigma, skew = np.full(2, 0.6), np.full(2, -0.25)
    mean = np.exp(np.log(response) + log_mean_loading(sigma, skew)[0])
    mean_eta = np.column_stack(
        (mean_links[0].link(mean), mean_links[1].link(sigma), mean_links[2].link(skew))
    )
    mean_plan = _plan(mean_family, response)
    # Only the scale moves, and mu = log m - log K(sigma, eps) moves with it.
    scale_only = np.tile([0.0, 1.0, 0.0], (2, 1))
    with pytest.raises(UnsupportedLikelihoodContractError, match="join"):
        mean_family.predictor_curvature_directional_derivative(
            response, mean_eta, scale_only, mean_links, mean_plan
        )


def test_tweedie_direction_differentiates_the_windowed_hessian(monkeypatch) -> None:
    """The analytic direction is the derivative of the published, windowed Hessian.

    At a deliberately short window (``log_cutoff = 6``) the window's distance
    from the full law is large, so the two candidate targets separate: the
    derivative of the windowed Hessian (its Richardson stencil, on rows whose
    window is the same at every stencil point) and the derivative of the full
    law (the direction at the default window).  On rows where the window is
    material the analytic direction sits at least ``1e4`` times nearer the
    first, near both power walls and between them.  This records that the
    series tail is the published likelihood's approximation, shared with the
    Hessian and with v0.40.0's stencil, not an error of the direction against
    the objective it differentiates.  It guards the shared window; it is
    evidence, not a fix, and passes at 7305fe25 as well.
    """
    import itertools

    import superglm.distributional.kernels.tweedie as tweedie_kernel
    from superglm.distributional.families import TweedieLSS

    family = TweedieLSS()
    links = _links(family)
    grid = itertools.product(
        (1.0505, 1.07, 1.5, 1.93, 1.9495), (0.05, 1.0, 10.0), (0.3, 5.0, 80.0), (1.0, 20.0)
    )
    power, dispersion, y, mean = np.array(list(grid)).T
    eta = np.column_stack((links[0].link(mean), links[1].link(dispersion), links[2].link(power)))
    plan = _plan(family, y)
    direction = np.random.default_rng(2).normal(size=eta.shape)
    full = family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    for function in (
        tweedie_kernel.tweedie_hessian_direction,
        tweedie_kernel.evaluate_tweedie_rows,
    ):
        monkeypatch.setitem(function.__kwdefaults__, "log_cutoff", 6.0)
    windowed = family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    step = 1.0e-4
    stencil = finite_difference_curvature_direction(
        family, y, eta, direction, links, plan, step=step
    )

    def window_terms(values):
        natural = np.column_stack([links[k].inverse(values[:, k]) for k in range(3)])
        return tweedie_kernel.evaluate_tweedie_rows(
            y, *natural.T, np.ones(len(y)), "prior", derivative_order=2
        ).terms

    unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    centre = window_terms(eta)
    same = np.ones(len(y), dtype=bool)
    for offset in (step, -step, step / 2, -step / 2):
        same &= window_terms(eta + offset * unit) == centre
    scale = np.max(np.abs(full), axis=1)
    to_stencil = np.max(np.abs(windowed - stencil.values), axis=1) / scale
    to_full = np.max(np.abs(windowed - full), axis=1) / scale
    material = same & (to_full > 1e-3)
    assert np.any(material & (power < 1.06)) and np.any(material & (power > 1.94))
    assert np.all(to_full[material] >= 1e4 * to_stencil[material])
