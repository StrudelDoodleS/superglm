"""Distribution functions for the Tweedie LSS atom and the prior-weighted families.

The Tweedie compound Poisson-gamma law on ``1 < p < 2`` has a point mass at zero
and a continuous part above it (Jorgensen 1987).  Its distribution function is
the Poisson sum ``F(y) = P(N = 0) + sum_k P(N = k) GammaCDF(y; k alpha, scale)``
truncated at a term whose omitted Poisson mass is below a stated tolerance; the
series density evaluation it is checked against here is Dunn & Smyth (2005).

The independent oracle in this file is the engine's own Tweedie kernel: the row
density is integrated on a log grid and compared with a difference of the
distribution function, at unit and at non-unit prior weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy import integrate, special, stats

from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.tweedie import TweedieLSS, _poisson_terms
from superglm.distributional.model import DenseDistributionalModel, fit_dense_distributional
from superglm.distributional.predictor import Predictor
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.weights import WeightContract
from superglm.features import Spline
from tests._distributional_family_kernels import tweedie as tweedie_kernel

# mean, dispersion, power, prior weight
_TWEEDIE_ROWS = (
    (0.5, 0.5, 1.20, 0.30),
    (1.0, 1.0, 1.50, 2.00),
    (2.0, 2.0, 1.70, 0.75),
    (5.0, 0.7, 1.50, 1.00),
    (0.2, 1.5, 1.90, 0.50),
)


def _tweedie_theta() -> NDArray[np.float64]:
    return np.array([row[:3] for row in _TWEEDIE_ROWS], dtype=np.float64)


def _tweedie_weights() -> NDArray[np.float64]:
    return np.array([row[3] for row in _TWEEDIE_ROWS], dtype=np.float64)


def _analytic_zero_mass(
    theta: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``P(Y = 0) = exp(-lambda)`` written straight from the compound rate."""
    mean, dispersion, power = theta[:, 0], theta[:, 1], theta[:, 2]
    tail_index = 2.0 - power
    return np.exp(-weights * mean**tail_index / (dispersion * tail_index))


def _kernel_density(
    grid: NDArray[np.float64], mean: float, dispersion: float, power: float, weight: float
) -> NDArray[np.float64]:
    """The engine kernel's own row density, used as the independent oracle."""
    ones = np.ones(len(grid), dtype=np.float64)
    evaluated = tweedie_kernel.evaluate_tweedie_rows(
        grid,
        mean * ones,
        dispersion * ones,
        power * ones,
        weight * ones,
        "prior",
        derivative_order=0,
    )
    assert np.all(evaluated.valid)
    return np.exp(evaluated.log_likelihood)


# ── the Tweedie atom and its distribution function ──────────────────────────


def test_tweedie_cdf_at_zero_is_the_poisson_void_probability() -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()
    zeros = np.zeros(len(theta))

    unit = family.cdf(zeros, theta)
    weighted = family.cdf_prior_weighted(zeros, theta, weights)

    assert unit == pytest.approx(_analytic_zero_mass(theta, np.ones(len(theta))), rel=1e-14)
    assert weighted == pytest.approx(_analytic_zero_mass(theta, weights), rel=1e-14)
    for index, (mean, dispersion, power, weight) in enumerate(_TWEEDIE_ROWS):
        kernel_mass = _kernel_density(np.zeros(1), mean, dispersion, power, weight)[0]
        assert weighted[index] == pytest.approx(kernel_mass, rel=1e-14)


def test_tweedie_cdf_is_monotone_and_reaches_one() -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()

    previous = np.zeros(len(theta))
    for value in (0.0, 1e-8, 1e-3, 0.1, 0.5, 1.0, 3.0, 12.0, 500.0, 1.0e6):
        current = family.cdf_prior_weighted(np.full(len(theta), value), theta, weights)
        assert np.all(current >= previous)
        previous = current
    assert np.all(previous > 1.0 - 1.0e-10)
    assert np.all(previous <= 1.0)
    assert np.all(family.cdf(np.full(len(theta), -1.0), theta) == 0.0)


def test_tweedie_cdf_left_limit_brackets_the_zero_atom() -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()
    zeros = np.zeros(len(theta))

    assert np.all(family.cdf_left_limit(zeros, theta) == 0.0)
    assert np.all(family.cdf_left_limit(zeros, theta, weights) == 0.0)
    assert np.all(family.cdf_left_limit(np.full(len(theta), -0.5), theta) == 0.0)

    positive = 0.75 * theta[:, 0]
    assert family.cdf_left_limit(positive, theta) == pytest.approx(
        family.cdf(positive, theta), rel=0.0, abs=0.0
    )
    assert family.cdf_left_limit(positive, theta, weights) == pytest.approx(
        family.cdf_prior_weighted(positive, theta, weights), rel=0.0, abs=0.0
    )
    # The atom interval a randomised PIT samples on a zero row is [0, P(Y = 0)].
    assert np.all(family.cdf_prior_weighted(zeros, theta, weights) > 0.0)


@pytest.mark.parametrize("factor", [0.05, 0.5, 1.0, 3.0])
def test_tweedie_quantile_inverts_the_cdf(factor: float) -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()
    response = factor * theta[:, 0]

    unit = family.cdf(response, theta)
    assert family.quantile(unit, theta) == pytest.approx(response, rel=1e-8)

    weighted = family.cdf_prior_weighted(response, theta, weights)
    assert family.quantile_prior_weighted(weighted, theta, weights) == pytest.approx(
        response, rel=1e-8
    )


def test_tweedie_quantile_returns_the_atom_at_and_below_the_zero_mass() -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()
    zero_mass = _analytic_zero_mass(theta, weights)

    assert np.all(family.quantile_prior_weighted(zero_mass, theta, weights) == 0.0)
    assert np.all(family.quantile_prior_weighted(0.5 * zero_mass, theta, weights) == 0.0)
    assert np.all(
        family.quantile(0.5 * _analytic_zero_mass(theta, np.ones(len(theta))), theta) == 0.0
    )
    assert np.all(
        family.quantile_prior_weighted(np.nextafter(zero_mass, 1.0), theta, weights) > 0.0
    )


def test_tweedie_quantile_rejects_probabilities_outside_the_open_unit_interval() -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()

    for probability in (0.0, 1.0):
        with pytest.raises(ValueError, match="strictly inside"):
            family.quantile(np.full(len(theta), probability), theta)
        with pytest.raises(ValueError, match="strictly inside"):
            family.quantile_prior_weighted(np.full(len(theta), probability), theta, weights)


@pytest.mark.parametrize(
    ("mean", "dispersion", "power", "weight"),
    [
        (0.5, 0.5, 1.20, 1.00),
        (1.0, 1.0, 1.50, 2.00),
        (2.0, 2.0, 1.70, 0.75),
        (0.2, 1.5, 1.90, 0.50),
    ],
)
def test_tweedie_cdf_matches_the_kernel_density_integral(
    mean: float, dispersion: float, power: float, weight: float
) -> None:
    """Integrate the kernel's row density on a log grid and difference the CDF."""
    family = TweedieLSS()
    theta = np.array([[mean, dispersion, power]], dtype=np.float64)
    weights = np.array([weight], dtype=np.float64)
    lower, upper = 0.02 * mean, 6.0 * mean

    exponent = np.linspace(np.log(lower), np.log(upper), 8001)
    grid = np.exp(exponent)
    # dy = y dt under y = exp(t): the log grid keeps the y^(alpha-1) edge smooth.
    integral = integrate.simpson(
        _kernel_density(grid, mean, dispersion, power, weight) * grid, x=exponent
    )

    difference = (
        family.cdf_prior_weighted(np.array([upper]), theta, weights)[0]
        - (family.cdf_prior_weighted(np.array([lower]), theta, weights)[0])
    )
    assert integral == pytest.approx(difference, rel=1e-6)


def test_tweedie_poisson_cap_accepts_a_certified_omitted_tail() -> None:
    omitted_tail = special.gammainc(20_001, 19_000.0)

    assert omitted_tail < 1.0e-12
    assert _poisson_terms(19_000.0) == 20_000


def test_tweedie_certified_cap_preserves_the_public_prior_weighted_law() -> None:
    family = TweedieLSS()
    powers = np.array([1.1, 1.5, 1.9], dtype=np.float64)
    weights = np.array([0.5, 2.0, 4.0], dtype=np.float64)
    effective_dispersion = 1.0 / (19_000.0 * (2.0 - powers))
    unit_theta = np.column_stack((np.ones(3), effective_dispersion, powers))
    weighted_theta = unit_theta.copy()
    weighted_theta[:, 1] *= weights
    response = np.array([0.99, 1.0, 1.02], dtype=np.float64)
    probabilities = np.array([0.25, 0.5, 0.75], dtype=np.float64)

    assert family.cdf_prior_weighted(response, weighted_theta, weights) == pytest.approx(
        family.cdf(response, unit_theta), rel=1e-13
    )
    assert family.quantile_prior_weighted(probabilities, weighted_theta, weights) == pytest.approx(
        family.quantile(probabilities, unit_theta), rel=1e-13
    )


@pytest.mark.parametrize("rate", [19_023.0, 1.0e300])
def test_tweedie_poisson_cap_refuses_an_uncertified_omitted_tail(rate: float) -> None:
    omitted_tail = float(special.gammainc(20_001, rate))
    assert omitted_tail > 1.0e-12

    with pytest.raises(ValueError) as exc_info:
        _poisson_terms(rate)

    message = str(exc_info.value)
    assert message.startswith("TweedieLSS refuses a compound Poisson rate")
    assert f"tail mass {omitted_tail:.3e}" in message
    assert "above the 1e-12 tolerance" in message


@pytest.mark.parametrize(
    "theta",
    [
        np.array([[1.0e6, 1.0e-3, 1.5]]),  # a finite rate above the term budget
        np.array([[1.0e300, 1.0e-300, 1.06]]),  # a rate that is not even finite
    ],
)
def test_tweedie_cdf_refuses_a_truncation_above_the_tail_tolerance(
    theta: NDArray[np.float64],
) -> None:
    family = TweedieLSS()
    # The second row overflows to an infinite rate on the way to the refusal.
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="tail mass"):
        family.cdf(np.array([1.0]), theta)


def test_tweedie_distribution_functions_validate_their_inputs() -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()

    with pytest.raises(ValueError, match="must be finite"):
        family.cdf(np.full(len(theta), np.nan), theta)
    with pytest.raises(ValueError, match="prior weights"):
        family.cdf_prior_weighted(np.zeros(len(theta)), theta, np.zeros(len(theta)))
    with pytest.raises(ValueError, match="prior weights"):
        family.quantile_prior_weighted(np.full(len(theta), 0.5), theta, -weights)
    with pytest.raises(ValueError, match="prior weights"):
        family.cdf_left_limit(np.zeros(len(theta)), theta, np.full(len(theta), np.inf))
    with pytest.raises(ValueError, match="outside its finite support"):
        family.cdf(np.zeros(1), np.array([[0.0, 1.0, 1.5]]))


# ── the fast quantile against the bisection it replaces ─────────────────────


def _wide_grid() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rows over the powers, four decades of mean and three of dispersion."""
    rows = [
        (mean, dispersion, power, weight)
        for power in (1.1, 1.3, 1.5, 1.7, 1.9)
        for mean in (0.02, 0.4, 8.0, 160.0)
        for dispersion in (0.1, 3.0, 100.0)
        for weight in (0.2, 1.0, 3.0)
    ]
    grid = np.array(rows, dtype=np.float64)
    return grid[:, :3], grid[:, 3]


def _wide_probabilities(
    theta: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Probabilities whose quantile is determined in float64 on every row."""
    atom = _analytic_zero_mass(theta, weights)
    return (
        np.full(len(theta), 1.0e-4),
        np.full(len(theta), 0.2),
        np.full(len(theta), 0.9),
        np.full(len(theta), 1.0 - 1.0e-6),
        atom + 0.5 * (1.0 - atom),
    )


def _atom_probabilities(
    theta: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Probabilities pinned to a row's atom or to one, where the root is a plateau.

    One ULP above ``P(Y = 0)`` the distribution function is flat over decades of
    ``y`` in float64, so no root finder can pin the quantile itself; what is
    determined there is the probability the answer lands on.
    """
    atom = _analytic_zero_mass(theta, weights)
    above = 1.0 - atom
    return (np.nextafter(atom, 1.0), atom + 1.0e-9 * above, 1.0 - 1.0e-6 * above)


def _untruncated_series_cdf(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    weights: NDArray[np.float64],
    terms: int,
) -> NDArray[np.float64]:
    """``P(Y <= y)`` summed from the first Poisson order with no window at all."""
    mean, dispersion, power = theta[:, 0], theta[:, 1], theta[:, 2]
    tail_index, jump_index = 2.0 - power, power - 1.0
    rate = weights * mean**tail_index / (dispersion * tail_index)
    jump_shape = tail_index / jump_index
    ratio = y / (dispersion * jump_index * mean**jump_index / weights)
    total = np.exp(-rate)
    for order in range(1, terms + 1):
        mass = np.exp(order * np.log(rate) - rate - special.gammaln(order + 1.0))
        total = total + mass * special.gammainc(order * jump_shape, ratio)
    return total


def test_tweedie_cdf_matches_an_untruncated_series_sum() -> None:
    """The windowed series drops only orders whose Poisson mass is negligible."""
    family = TweedieLSS()
    theta = np.array(
        [
            [0.5, 0.5, 1.20],
            [2.0, 2.0, 1.70],
            [0.2, 1.5, 1.90],
            [40.0, 0.05, 1.30],  # rate ~ 1.6e3, so the window starts well above one
        ],
        dtype=np.float64,
    )
    weights = np.array([0.3, 0.75, 0.5, 1.0])
    for factor in (0.1, 0.5, 1.0, 2.0, 10.0):
        response = factor * theta[:, 0]
        windowed = family.cdf_prior_weighted(response, theta, weights)
        reference = _untruncated_series_cdf(response, theta, weights, terms=4000)
        assert windowed == pytest.approx(reference, rel=0.0, abs=1e-12)


def test_tweedie_quantile_matches_the_bisection_it_replaces() -> None:
    """The Householder root finder inverts exactly the function the bisection did."""
    family = TweedieLSS()
    theta, weights = _wide_grid()
    for probabilities in _wide_probabilities(theta, weights):
        fast = family.quantile_prior_weighted(probabilities, theta, weights)
        reference = family._quantile_bisection(probabilities, theta, weights)
        assert np.array_equal(fast == 0.0, reference == 0.0)
        positive = fast > 0.0
        assert positive.sum() > 40
        assert fast[positive] == pytest.approx(reference[positive], rel=1e-10)


def test_tweedie_quantile_inverts_the_atom_edge_as_the_bisection_does() -> None:
    """On the plateau just above the atom both land on the same probability."""
    family = TweedieLSS()
    theta, weights = _wide_grid()
    for probabilities in _atom_probabilities(theta, weights):
        fast = family.quantile_prior_weighted(probabilities, theta, weights)
        reference = family._quantile_bisection(probabilities, theta, weights)
        assert np.array_equal(fast == 0.0, reference == 0.0)
        positive = fast > 0.0
        assert positive.sum() > 100
        reached = family.cdf_prior_weighted(fast, theta, weights)[positive]
        assert reached == pytest.approx(probabilities[positive], rel=1e-14)
        assert reached == pytest.approx(
            family.cdf_prior_weighted(reference, theta, weights)[positive], rel=1e-14
        )


def test_tweedie_quantile_holds_the_atom_rule_over_the_wide_grid() -> None:
    family = TweedieLSS()
    theta, weights = _wide_grid()
    atom = _analytic_zero_mass(theta, weights)
    # Rows whose atom has underflowed to zero or rounded to one carry no
    # representable probability on the atom to ask about.
    interior = (atom > 0.0) & (atom < 1.0 - 1.0e-12)
    assert interior.sum() > 100
    theta, weights, atom = theta[interior], weights[interior], atom[interior]

    assert np.all(family.quantile_prior_weighted(atom, theta, weights) == 0.0)
    assert np.all(family.quantile_prior_weighted(0.25 * atom, theta, weights) == 0.0)
    assert np.all(family.quantile_prior_weighted(np.nextafter(atom, 1.0), theta, weights) > 0.0)


def test_tweedie_quantile_is_monotone_in_the_probability() -> None:
    family = TweedieLSS()
    theta, weights = _wide_grid()
    atom = _analytic_zero_mass(theta, weights)
    previous = np.zeros(len(theta))
    for level in (1.0e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1.0e-6):
        current = family.quantile_prior_weighted(atom + level * (1.0 - atom), theta, weights)
        assert np.all(current >= previous)
        previous = current


def test_tweedie_quantile_round_trips_the_distribution_function() -> None:
    family = TweedieLSS()
    theta, weights = _wide_grid()
    for factor in (0.05, 0.3, 1.0, 4.0, 40.0):
        response = factor * theta[:, 0]
        probabilities = family.cdf_prior_weighted(response, theta, weights)
        atom = _analytic_zero_mass(theta, weights)
        # Below about 1e-6 of the conditional mass the probability itself no
        # longer resolves the response, and neither does 1 - p at the other end:
        # both differences are float64 rounding there.
        conditional = (probabilities - atom) / (1.0 - atom)
        reachable = (conditional > 1.0e-6) & (conditional < 1.0 - 1.0e-7)
        bulk = (conditional > 1.0e-3) & (conditional < 1.0 - 1.0e-3)
        assert bulk.sum() > 50
        recovered = family.quantile_prior_weighted(
            probabilities[reachable], theta[reachable], weights[reachable]
        )
        assert recovered == pytest.approx(response[reachable], rel=1e-5)
        assert recovered[bulk[reachable]] == pytest.approx(response[bulk], rel=1e-9)


def test_tweedie_quantile_refuses_exactly_what_the_bisection_refuses() -> None:
    family = TweedieLSS()
    theta = _tweedie_theta()
    weights = _tweedie_weights()
    probabilities = np.full(len(theta), 0.5)

    for bad in (np.zeros(len(theta)), np.ones(len(theta))):
        with pytest.raises(ValueError, match="strictly inside"):
            family.quantile_prior_weighted(bad, theta, weights)
        with pytest.raises(ValueError, match="strictly inside"):
            family._quantile_bisection(bad, theta, weights)
    with pytest.raises(ValueError, match="prior weights"):
        family._quantile_bisection(probabilities, theta, np.zeros(len(theta)))
    refused = np.array([[1.0e6, 1.0e-3, 1.5]])
    for call in (family.quantile_prior_weighted, family._quantile_bisection):
        with pytest.raises(ValueError, match="tail mass"):
            call(np.array([0.5]), refused, np.array([1.0]))


# ── the prior-weighted trio ─────────────────────────────────────────────────


def _gaussian_theta() -> NDArray[np.float64]:
    return np.array([[0.0, 1.0], [-2.0, 0.5], [3.5, 2.0]], dtype=np.float64)


def _gamma_theta() -> NDArray[np.float64]:
    return np.array([[1.0, 0.5], [4.0, 1.2], [0.3, 0.8]], dtype=np.float64)


def test_gaussian_prior_weighted_functions_match_scipy() -> None:
    family = GaussianLS()
    theta = _gaussian_theta()
    weights = np.array([0.25, 1.5, 4.0])
    response = np.array([0.4, -1.0, 5.0])
    probabilities = np.array([0.05, 0.5, 0.975])
    scale = theta[:, 1] / np.sqrt(weights)

    assert family.cdf_prior_weighted(response, theta, weights) == pytest.approx(
        stats.norm.cdf(response, loc=theta[:, 0], scale=scale), rel=1e-13
    )
    assert family.quantile_prior_weighted(probabilities, theta, weights) == pytest.approx(
        stats.norm.ppf(probabilities, loc=theta[:, 0], scale=scale), rel=1e-12
    )


def test_gamma_prior_weighted_functions_match_scipy() -> None:
    family = GammaLS()
    theta = _gamma_theta()
    weights = np.array([0.25, 1.5, 4.0])
    response = np.array([0.7, 3.0, 0.2])
    probabilities = np.array([0.05, 0.5, 0.975])
    squared_cv = theta[:, 1] ** 2
    shape = weights / squared_cv
    scale = theta[:, 0] * squared_cv / weights

    assert family.cdf_prior_weighted(response, theta, weights) == pytest.approx(
        stats.gamma.cdf(response, a=shape, scale=scale), rel=1e-12
    )
    assert family.quantile_prior_weighted(probabilities, theta, weights) == pytest.approx(
        stats.gamma.ppf(probabilities, a=shape, scale=scale), rel=1e-10
    )


def test_prior_weight_of_one_reproduces_the_unweighted_functions() -> None:
    probabilities = np.array([0.05, 0.5, 0.975])
    cases = (
        (GaussianLS(), _gaussian_theta(), np.array([0.4, -1.0, 5.0])),
        (GammaLS(), _gamma_theta(), np.array([0.7, 3.0, 0.2])),
        (TweedieLSS(), _tweedie_theta(), 0.8 * _tweedie_theta()[:, 0]),
    )
    for family, theta, response in cases:
        unit = np.ones(len(theta))
        probability_vector = np.resize(probabilities, len(theta))
        assert family.cdf_prior_weighted(response, theta, unit) == pytest.approx(
            family.cdf(response, theta), rel=1e-14
        )
        assert family.quantile_prior_weighted(probability_vector, theta, unit) == pytest.approx(
            family.quantile(probability_vector, theta), rel=1e-12
        )


@pytest.mark.parametrize("family", [GaussianLS(), GammaLS()])
def test_prior_weighted_functions_validate_their_inputs(family: object) -> None:
    theta = _gaussian_theta() if isinstance(family, GaussianLS) else _gamma_theta()
    weights = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="prior weights"):
        family.cdf_prior_weighted(np.zeros(len(theta)), theta, np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="prior weights"):
        family.quantile_prior_weighted(np.full(len(theta), 0.5), theta, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="strictly inside"):
        family.quantile_prior_weighted(np.zeros(len(theta)), theta, weights)


# ── the weight matters on a fitted burn-cost model ──────────────────────────


def _simulated_burn_cost(
    n_rows: int = 1500, seed: int = 20260903
) -> tuple[pd.DataFrame, NDArray[np.float64], NDArray[np.float64]]:
    """Compound Poisson-gamma rows drawn without a production Tweedie evaluator."""
    rng = np.random.default_rng(seed)
    first = rng.uniform(-1.0, 1.0, n_rows)
    second = rng.uniform(-1.0, 1.0, n_rows)
    weights = rng.uniform(0.2, 1.0, n_rows)
    mean = np.exp(0.4 + 0.5 * np.sin(np.pi * first))
    dispersion = np.exp(-0.2 + 0.3 * second)
    power = np.full(n_rows, 1.5)
    tail_index, jump_index = 2.0 - power, power - 1.0
    rate = weights * mean**tail_index / (dispersion * tail_index)
    jump_scale = dispersion * jump_index * mean**jump_index / weights
    counts = rng.poisson(rate)
    response = np.zeros(n_rows, dtype=np.float64)
    claimed = counts > 0
    response[claimed] = rng.gamma(
        (tail_index / jump_index)[claimed] * counts[claimed], jump_scale[claimed]
    )
    return pd.DataFrame({"first": first, "second": second}), response, weights


@pytest.fixture(scope="module")
def burn_cost_fit() -> tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray]:
    frame, response, weights = _simulated_burn_cost()
    model = fit_dense_distributional(
        frame,
        response,
        family=TweedieLSS(),
        predictors=(
            Predictor("mean", {"first": Spline(kind="cr", n_knots=6)}),
            Predictor("dispersion", {"second": Spline(kind="cr", n_knots=5)}),
            Predictor("power", {}),
        ),
        weight_contract=WeightContract("prior"),
        sample_weight=weights,
        config=DenseSolverConfig(coefficient_curvature="observed", tolerance=1.0e-8),
        lambdas={"mean:first#wiggle": 1.0, "dispersion:second#wiggle": 1.0},
        discrete=False,
        chunk_size=None,
    )
    return model, frame, response, weights


def test_prior_weighted_pit_is_uniform_and_the_unit_weight_pit_is_not(
    burn_cost_fit: tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray],
) -> None:
    """The exposure weight is part of the row's law; dropping it breaks the PIT."""
    model, frame, response, weights = burn_cost_fit
    family = TweedieLSS()
    theta = model.predict_parameters(frame)
    claimed = response > 0.0
    assert claimed.sum() > 1000

    def conditional_pit(row_weights: NDArray[np.float64]) -> NDArray[np.float64]:
        distribution = family.cdf_prior_weighted(response, theta, row_weights)
        zero_mass = family.cdf_prior_weighted(np.zeros(len(response)), theta, row_weights)
        return (distribution[claimed] - zero_mass[claimed]) / (1.0 - zero_mass[claimed])

    weighted = conditional_pit(weights)
    unit = conditional_pit(np.ones(len(response)))

    assert np.all((weighted > 0.0) & (weighted < 1.0))
    assert stats.kstest(weighted, "uniform").pvalue > 0.01
    assert stats.kstest(unit, "uniform").pvalue < 1.0e-3


def test_prior_weighted_quantile_reproduces_the_fitted_response(
    burn_cost_fit: tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray],
) -> None:
    model, frame, response, weights = burn_cost_fit
    family = TweedieLSS()
    theta = model.predict_parameters(frame)
    claimed = np.flatnonzero(response > 0.0)[:40]

    probabilities = family.cdf_prior_weighted(response, theta, weights)[claimed]
    recovered = family.quantile_prior_weighted(probabilities, theta[claimed], weights[claimed])
    assert recovered == pytest.approx(response[claimed], rel=1e-8)
