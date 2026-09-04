"""Tweedie location-scale-shape family for dense distributional fitting."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special
from scipy.special import expit
from scipy.stats import poisson

from superglm.distributional.families._base import (
    readonly,
    typed_plan,
    validated_float_response,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    FamilyCapabilities,
    FamilyLikelihoodPlan,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
    _validated_derivative_order,
    _validated_parameter_matrix,
)
from superglm.distributional.kernels.tweedie import (
    TweedieNumericalRefusal,
    evaluate_tweedie_rows,
    initialize_tweedie,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import Link, LogLink


def _finite_wall(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("power walls must be finite and strictly ordered inside (1, 2)")
    try:
        wall = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("power walls must be finite and strictly ordered inside (1, 2)") from exc
    if not math.isfinite(wall):
        raise ValueError(f"{name} power wall must be finite")
    return wall


@dataclass(frozen=True)
class BoundedPowerLink:
    """Logit link from an open configured power interval to the real line."""

    lower: float = 1.05
    upper: float = 1.95

    def __post_init__(self) -> None:
        lower = _finite_wall(self.lower, name="lower")
        upper = _finite_wall(self.upper, name="upper")
        if not 1.0 < lower < upper < 2.0:
            raise ValueError("power walls must be strictly ordered inside (1, 2)")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def _interior_values(self, value: NDArray, *, name: str) -> NDArray[np.float64]:
        try:
            values = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} input must be finite and strictly between power walls"
            ) from exc
        if (
            not np.all(np.isfinite(values))
            or np.any(values <= self.lower)
            or np.any(values >= self.upper)
        ):
            raise ValueError(f"{name} input must be finite and strictly between power walls")
        return values

    def link(self, mu: NDArray) -> NDArray[np.float64]:
        values = self._interior_values(mu, name="link")
        return np.log(values - self.lower) - np.log(self.upper - values)

    def inverse(self, eta: NDArray) -> NDArray[np.float64]:
        probability = expit(np.asarray(eta, dtype=np.float64))
        return self.lower + (self.upper - self.lower) * probability

    def deriv(self, mu: NDArray) -> NDArray[np.float64]:
        values = self._interior_values(mu, name="derivative")
        return (self.upper - self.lower) / ((values - self.lower) * (self.upper - values))

    def deriv_inverse(self, eta: NDArray) -> NDArray[np.float64]:
        probability = expit(np.asarray(eta, dtype=np.float64))
        return (self.upper - self.lower) * probability * (1.0 - probability)

    def deriv2_inverse(self, eta: NDArray) -> NDArray[np.float64]:
        probability = expit(np.asarray(eta, dtype=np.float64))
        first = (self.upper - self.lower) * probability * (1.0 - probability)
        return first * (1.0 - 2.0 * probability)


_POISSON_TAIL_TOLERANCE = 1.0e-12
_MAX_POISSON_TERMS = 20_000
_QUANTILE_BISECTIONS = 64
# ``exp`` of this pair spans every positive float64: below the first the
# distribution function is its zero mass and above the last it is one, so the
# pair brackets the quantile of any row without a search.
_LOG_QUANTILE_LOWER = -745.0
_LOG_QUANTILE_UPPER = 709.0
# Each end of a row's series window may omit half the tail tolerance.
_SERIES_TAIL_LEVEL = -math.log(0.5 * _POISSON_TAIL_TOLERANCE)
_SERIES_WINDOW_STEPS = 8
_SERIES_WINDOW_FLOOR = 1.0e-12
# The term axis of one series slab, in elements: a handful of arrays this size
# are alive while a slab is solved.
_SERIES_BLOCK_ELEMENTS = 1_000_000
# How much wider than its narrowest row a slab's term axis may be: the waste of
# one more slab against the waste of evaluating terms no row of it needs.
_SERIES_WIDTH_SPREAD = 1.1
_SERIES_WIDTH_SLACK = 2.0
# Below this share of the conditional mass the single-jump term carries the
# distribution function on its own, and its own gamma is the better start.
_SINGLE_JUMP_SHARE = 0.3
# A Householder step of order three converges quartically, so a step this small
# leaves a root error far below the float64 resolution of the answer.
_QUANTILE_STEP_TOLERANCE = 1.0e-5
_QUANTILE_BRACKET_TOLERANCE = 1.0e-13
# Enough for a row that never takes a Householder step to bisect the whole
# bracket down to the bracket tolerance; the loop leaves as soon as every row
# has settled, so the cap costs nothing on the rows that converge in three.
_QUANTILE_ITERATIONS = 200
_TINY_PROBABILITY = 1.0e-300
_PROBABILITY_MARGIN = 1.0e-16


def _prior_weight_vector(weights: NDArray, n_observations: int) -> NDArray[np.float64]:
    """Per-row prior weights broadcast to the parameter rows of one call."""
    values = np.broadcast_to(np.asarray(weights, dtype=np.float64), (n_observations,))
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("prior weights must be finite and strictly positive")
    return values


def _compound_poisson_parameters(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The compound Poisson-gamma rate, jump shape and jump scale of each row.

    A Tweedie row on ``1 < p < 2`` is the total of ``N ~ Poisson(rate)``
    independent ``Gamma(jump_shape, jump_scale)`` jumps (Jorgensen 1987).  A
    prior weight is the exponential-dispersion weight: it enters the row's law
    as the dispersion ``phi / w``, which scales the rate up and the jump scale
    down while leaving the mean where it is.
    """
    mean, dispersion, power = values[:, 0], values[:, 1], values[:, 2]
    tail_index = 2.0 - power
    jump_index = power - 1.0
    rate = weights * mean**tail_index / (dispersion * tail_index)
    jump_shape = tail_index / jump_index
    jump_scale = dispersion * jump_index * mean**jump_index / weights
    return rate, jump_shape, jump_scale


def _poisson_terms(rate_max: float) -> int:
    """The truncation point whose omitted Poisson mass is below the tail tolerance.

    Every omitted term contributes at most its own Poisson mass to the mixture,
    so the tail mass bounds the truncation error of the distribution function.
    """
    if not math.isfinite(rate_max):
        raise ValueError(
            f"TweedieLSS refuses a compound Poisson rate of {rate_max:.6g}: the tail "
            f"mass beyond {_MAX_POISSON_TERMS} terms cannot be certified for a "
            "non-finite rate"
        )
    if rate_max < _MAX_POISSON_TERMS:
        floor_terms = max(5, int(rate_max + 8.0 * math.sqrt(max(rate_max, 1.0))) + 1)
        tail_terms = float(poisson.isf(_POISSON_TAIL_TOLERANCE, rate_max))
        if math.isfinite(tail_terms):
            terms = max(floor_terms, int(tail_terms))
            if terms <= _MAX_POISSON_TERMS:
                return terms
    tail = float(special.gammainc(_MAX_POISSON_TERMS + 1, rate_max))
    if math.isfinite(tail) and tail <= _POISSON_TAIL_TOLERANCE:
        return _MAX_POISSON_TERMS
    raise ValueError(
        f"TweedieLSS refuses a compound Poisson rate of {rate_max:.6g}: truncating the "
        f"series at {_MAX_POISSON_TERMS} terms leaves tail mass {tail:.3e}, above the "
        f"{_POISSON_TAIL_TOLERANCE:.0e} tolerance"
    )


def _series_window(
    rate: NDArray[np.float64], terms: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The Poisson orders a row keeps; the rest carry mass below the tolerance.

    Dunn & Smyth (2005, Section 4) sum a Tweedie series only where its terms
    live rather than from index one, because "the naive approach of starting at
    index 1 and adding more terms can result in a very large number of
    unnecessary terms".  Every term of this series is bounded by its own Poisson
    mass, so the window is where Chernoff's bound on a Poisson count first falls
    below half the tail tolerance on each side: with
    ``D(k) = k log(k / rate) - k + rate`` both ``P(N >= k)`` for ``k > rate`` and
    ``P(N <= k)`` for ``k < rate`` are at most ``exp(-D(k))``.  ``D`` is convex
    with derivative ``log(k / rate)``, so Newton's method from a start outside
    the root converges to it monotonically from that side.
    """
    level = _SERIES_TAIL_LEVEL
    spread = np.sqrt(2.0 * level * rate)
    upper = rate + spread + level
    for _ in range(_SERIES_WINDOW_STEPS):
        slope = np.log(upper / rate)
        upper = np.maximum(
            upper - (upper * slope - upper + rate - level) / slope, rate * (1.0 + 1.0e-6)
        )
    highest = np.clip(np.ceil(upper), 1.0, float(terms))
    reaches = rate > level
    if not np.any(reaches):
        return np.ones_like(highest), highest
    lower = np.clip(rate - spread - level, _SERIES_WINDOW_FLOOR, rate * (1.0 - 1.0e-6))
    for _ in range(_SERIES_WINDOW_STEPS):
        slope = np.log(lower / rate)
        step = (lower * slope - lower + rate - level) / slope
        lower = np.clip(lower - step, _SERIES_WINDOW_FLOOR, rate * (1.0 - 1.0e-6))
    lowest = np.clip(np.floor(np.where(reaches, lower, 0.0)) + 1.0, 1.0, highest)
    return lowest, highest


@dataclass(frozen=True)
class _SeriesBlock:
    """Rows sharing one Poisson term axis, terms first, rows second.

    Everything here depends on the row's parameters alone and not on where the
    series is evaluated, so one block is built once and reused by every
    iteration of a quantile solve.
    """

    shape: NDArray[np.float64]
    log_gamma_shape: NDArray[np.float64] | None
    mass: NDArray[np.float64]
    scale: NDArray[np.float64]
    atom: NDArray[np.float64]
    rate: NDArray[np.float64]
    jump_shape: NDArray[np.float64]

    def take(self, columns: NDArray[np.intp]) -> _SeriesBlock:
        gamma_shape = self.log_gamma_shape
        return _SeriesBlock(
            self.shape[:, columns],
            None if gamma_shape is None else gamma_shape[:, columns],
            self.mass[:, columns],
            self.scale[columns],
            self.atom[columns],
            self.rate[columns],
            self.jump_shape[columns],
        )


def _build_series_block(
    rate: NDArray[np.float64],
    jump_shape: NDArray[np.float64],
    jump_scale: NDArray[np.float64],
    lowest: NDArray[np.float64],
    highest: NDArray[np.float64],
    derivatives: bool,
) -> _SeriesBlock:
    width = int(np.max(highest - lowest)) + 1
    steps = np.arange(width, dtype=np.float64)[:, None]
    # When no row's window starts above the first order the orders are the same
    # column for every row, and the log factorial of the Poisson mass is one
    # column rather than a whole slab.
    order = 1.0 + steps if np.all(lowest == 1.0) else lowest + steps
    inside = order <= highest
    mass = np.where(inside, np.exp(order * np.log(rate) - rate - special.gammaln(order + 1.0)), 0.0)
    atom = np.exp(-rate)
    if np.any(lowest > 1.0):
        # The dropped orders below the window carry a regularised incomplete
        # gamma no smaller than the window's first order, whose shape they sit
        # under, so carrying their whole Poisson mass on that first term keeps
        # the sum a lower bound on the untruncated series and shrinks its
        # deficit below that mass.
        mass[0] += special.gammaincc(lowest, rate) - atom
    shape = order * jump_shape
    # Only the density reads the log gamma of every term's shape; a lone
    # distribution-function call would pay for it and never use it.
    log_gamma_shape = special.gammaln(shape) if derivatives else None
    return _SeriesBlock(shape, log_gamma_shape, mass, jump_scale, atom, rate, jump_shape)


def _series_blocks(
    rate: NDArray[np.float64],
    jump_shape: NDArray[np.float64],
    jump_scale: NDArray[np.float64],
    rows: NDArray[np.intp],
    terms: int,
    *,
    derivatives: bool = False,
) -> Iterator[tuple[NDArray[np.intp], _SeriesBlock]]:
    """Slabs of ``(rows, block)`` sharing a term axis of nearly one width.

    A slab evaluates its whole term axis on every row it holds, so rows are
    sorted by window width and cut into slabs whose widest row is within
    ``_SERIES_WIDTH_SPREAD`` of its narrowest.  Without that cut one slab of
    mixed widths would evaluate the widest row's term count on every row and
    give back the whole point of the per-row window.  The element budget cuts
    again where a slab would be too large to hold.
    """
    lowest, highest = _series_window(rate[rows], terms)
    widths = highest - lowest + 1.0
    ordering = np.argsort(widths, kind="stable")
    rows, lowest, highest, widths = (
        rows[ordering],
        lowest[ordering],
        highest[ordering],
        widths[ordering],
    )
    start = 0
    while start < len(rows):
        spread = widths[start] * _SERIES_WIDTH_SPREAD + _SERIES_WIDTH_SLACK
        stop = max(int(np.searchsorted(widths, spread, side="right")), start + 1)
        width = int(widths[stop - 1])
        stop = min(stop, start + max(1, _SERIES_BLOCK_ELEMENTS // width))
        chunk = rows[start:stop]
        yield (
            chunk,
            _build_series_block(
                rate[chunk],
                jump_shape[chunk],
                jump_scale[chunk],
                lowest[start:stop],
                highest[start:stop],
                derivatives,
            ),
        )
        start = stop


def _block_distribution(block: _SeriesBlock, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """``P(Y <= y)`` on the block's rows: one incomplete gamma per kept term."""
    ratio = y / block.scale
    total = block.atom + np.einsum("ij,ij->j", block.mass, special.gammainc(block.shape, ratio))
    return np.minimum(total, 1.0)


def _block_log_derivatives(
    block: _SeriesBlock, y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """``F`` and its first three derivatives in ``log y``, from one series pass.

    With ``r = y / scale`` the term ``mass * r**a exp(-r) / Gamma(a)`` is
    ``y f(y)`` order by order, and differentiating it in ``log y`` multiplies it
    by ``a - r``; the three derivatives therefore cost two more passes over the
    terms the distribution function already needed and no further special
    functions.
    """
    ratio = y / block.scale
    total = np.minimum(
        block.atom + np.einsum("ij,ij->j", block.mass, special.gammainc(block.shape, ratio)), 1.0
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        term = block.mass * np.exp(block.shape * np.log(ratio) - ratio - block.log_gamma_shape)
        centred = block.shape - ratio
        first = term.sum(axis=0)
        # Weighting before squaring keeps a vanished term at zero instead of
        # multiplying it by an overflowing square out in the far tail.
        weighted = term * centred
        second = weighted.sum(axis=0)
        third = np.einsum("ij,ij->j", weighted, centred) - ratio * first
    return total, first, second, third


def _block_start(block: _SeriesBlock, target: NDArray[np.float64]) -> NDArray[np.float64]:
    """``log q`` from a gamma matched to the row's conditional positive part.

    A gamma matched to the first two moments of an aggregate compound sum is the
    standard approximation of actuarial practice (Hardy 2004); the moments of
    the sum conditioned on at least one jump are closed forms of the compound
    rate, jump shape and jump scale.  Deep in the left tail the single-jump term
    carries the distribution function alone, and there the exact
    ``Gamma(jump shape, jump scale)`` tail is the better start.
    """
    survive = -np.expm1(-block.rate)
    mean = block.rate * block.jump_shape * block.scale
    variance = block.rate * block.jump_shape * (1.0 + block.jump_shape) * block.scale**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        conditional_mean = mean / survive
        # ``1 - survive`` is the atom exactly, so this form leaves the
        # correction on the atom rather than on a difference of two nearly
        # equal squares.
        conditional_variance = np.maximum(
            variance / survive - mean * mean * block.atom / (survive * survive),
            _TINY_PROBABILITY,
        )
        conditional = np.clip(
            (target - block.atom) / survive, _TINY_PROBABILITY, 1.0 - _PROBABILITY_MARGIN
        )
        single = block.rate * block.atom / survive
        share = np.where(single > 0.0, conditional / single, np.inf)
        lone = share < _SINGLE_JUMP_SHARE
        shape = np.where(lone, block.jump_shape, conditional_mean**2 / conditional_variance)
        scale = np.where(lone, block.scale, conditional_variance / conditional_mean)
        level = np.where(
            lone, np.clip(share, _TINY_PROBABILITY, 1.0 - _PROBABILITY_MARGIN), conditional
        )
    return np.clip(
        np.log(np.maximum(special.gammaincinv(shape, level) * scale, _TINY_PROBABILITY)),
        _LOG_QUANTILE_LOWER + 1.0,
        _LOG_QUANTILE_UPPER - 1.0,
    )


def _block_quantile(block: _SeriesBlock, target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve ``F(q) = target`` on ``log q`` by a bracketed Householder iteration.

    Newton's iteration on ``F(q) - p`` converges monotonically for any
    continuous unimodal distribution when it starts at the mode, because the
    distribution function is convex below the mode and concave above it (Giner &
    Smyth 2016).  A Tweedie density is not always unimodal: it is multimodal for
    ``p`` near one and unbounded at zero for ``p`` above 1.5 (Dunn & Smyth 2005,
    Section 8), both inside this family's power walls.  The iteration is
    therefore bracketed and falls back to bisection whenever a step leaves the
    bracket or stops reducing the residual fast enough, which is the ``rtsafe``
    safeguard of Press et al. (2007, Section 9.4) and leaves the bracket the
    guarantee of convergence.  The step is Householder's third-order form, whose
    two extra derivatives come free from the series pass the residual needed.
    """
    rows = len(target)
    position = _block_start(block, target)
    lower = np.full(rows, _LOG_QUANTILE_LOWER)
    upper = np.full(rows, _LOG_QUANTILE_UPPER)
    previous = np.full(rows, _LOG_QUANTILE_UPPER - _LOG_QUANTILE_LOWER)
    answer = np.exp(position)
    active = np.arange(rows)
    for _ in range(_QUANTILE_ITERATIONS):
        if active.size == 0:
            break
        current = position[active]
        distribution, first, second, third = _block_log_derivatives(
            block.take(active), np.exp(current)
        )
        residual = distribution - target[active]
        beneath = residual < 0.0
        low = np.where(beneath, current, lower[active])
        high = np.where(beneath, upper[active], current)
        lower[active], upper[active] = low, high
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = (
                6.0 * first**3 - 6.0 * residual * first * second + residual * residual * third
            )
            numerator = 6.0 * residual * first * first - 3.0 * residual * residual * second
            step = np.where(denominator != 0.0, -numerator / denominator, -residual / first)
            proposed = current + step
            halve = (
                ~np.isfinite(proposed)
                | (proposed < low)
                | (proposed > high)
                | (np.abs(2.0 * residual) > np.abs(previous[active] * first))
            )
        proposed = np.where(halve, 0.5 * (low + high), proposed)
        previous[active] = np.abs(proposed - current)
        position[active] = proposed
        answer[active] = np.exp(proposed)
        magnitude = np.maximum(np.abs(current), 1.0)
        settled = (
            ~halve & (np.abs(proposed - current) <= _QUANTILE_STEP_TOLERANCE * magnitude)
        ) | (high - low <= _QUANTILE_BRACKET_TOLERANCE * magnitude)
        active = active[~settled]
    return answer


def _validated_probabilities(p: NDArray, n_observations: int) -> NDArray[np.float64]:
    probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (n_observations,))
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
    return probabilities


_CAPABILITIES = FamilyCapabilities(
    max_derivative_order=2,
    expected_information=False,
    cdf=True,
    quantile=True,
    random=False,
    response_mean=True,
)


@dataclass(frozen=True)
class TweedieLikelihoodPlan:
    """The resolved positional weights bound to a Tweedie fit."""

    weights: ResolvedLikelihoodWeights

    def __post_init__(self) -> None:
        if not isinstance(self.weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "Tweedie likelihood plan requires resolved likelihood weights"
            )

    @property
    def plan_identifier(self) -> str:
        return f"TweedieLSS/v1:{self.weights.digest}"

    def take(self, indices: NDArray[np.integer]) -> TweedieLikelihoodPlan:
        return type(self)(weights=self.weights.take(indices))


def _validated_response(y: NDArray) -> NDArray[np.float64]:
    return validated_float_response(
        y, message="y must be a non-empty finite non-negative vector", lower=0.0
    )


def _validated_plan(plan: FamilyLikelihoodPlan, *, n_observations: int) -> TweedieLikelihoodPlan:
    return typed_plan(plan, TweedieLikelihoodPlan, n_observations, family_name="TweedieLSS")


@dataclass(frozen=True)
class TweedieLSS:
    """Three-parameter normalized Tweedie family on configured interior walls."""

    power_lower: float = 1.05
    power_upper: float = 1.95

    def __post_init__(self) -> None:
        link = BoundedPowerLink(self.power_lower, self.power_upper)
        object.__setattr__(self, "power_lower", link.lower)
        object.__setattr__(self, "power_upper", link.upper)

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                "mean",
                LogLink(),
                "mean",
                ParameterSupport(lower=0.0),
                "observed",
            ),
            ParameterSpec(
                "dispersion",
                LogLink(),
                "dispersion",
                ParameterSupport(lower=0.0),
                "observed",
            ),
            ParameterSpec(
                "power",
                BoundedPowerLink(self.power_lower, self.power_upper),
                "power",
                ParameterSupport(lower=self.power_lower, upper=self.power_upper),
                "observed",
            ),
        )

    @property
    def default_prediction_name(self) -> str:
        return "conditional_mean"

    @property
    def capabilities(self) -> FamilyCapabilities:
        return _CAPABILITIES

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "TweedieLSS",
            "power_lower": self.power_lower,
            "power_upper": self.power_upper,
        }

    def response_boundaries(self, links: Sequence[Link]) -> tuple[tuple[str, ...], ...]:
        """All-zero rows drive the mean to 0 (log link) or the dispersion to infinity.

        With every response at zero the row log-likelihood is
        ``-w mu^(2-p) / (phi (2-p))``, which increases without bound as
        ``log mu -> -inf`` or ``log phi -> +inf``; the power predictor is
        bounded on its interior walls.
        """
        mean_link, dispersion_link, _power_link = tuple(links)
        return (
            ("zero",) if isinstance(mean_link, LogLink) else (),
            ("zero",) if isinstance(dispersion_link, LogLink) else (),
            (),
        )

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> TweedieLikelihoodPlan:
        response = _validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "TweedieLSS supports only the complete-observation contract"
            )
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "TweedieLSS requires resolved likelihood weights"
            )
        if len(weights.values) != len(response):
            raise UnsupportedLikelihoodContractError(
                "Tweedie response rows do not match resolved likelihood-weight rows"
            )
        return TweedieLikelihoodPlan(weights=weights)

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = _validated_response(y)
        tweedie_plan = _validated_plan(plan, n_observations=len(response))
        theta = initialize_tweedie(
            response,
            tweedie_plan.weights.values,
            tweedie_plan.weights.provenance.contract.semantics,
            power_lower=self.power_lower,
            power_upper=self.power_upper,
        )
        return InitialParameterState(theta=theta)

    def evaluate_natural(
        self,
        y: NDArray,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        order = _validated_derivative_order(derivative_order)
        response = _validated_response(y)
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=len(response),
            parameters=self.parameters,
            family_name="TweedieLSS",
        )
        tweedie_plan = _validated_plan(plan, n_observations=len(response))
        try:
            evaluated = evaluate_tweedie_rows(
                response,
                parameters[:, 0],
                parameters[:, 1],
                parameters[:, 2],
                tweedie_plan.weights.values,
                tweedie_plan.weights.provenance.contract.semantics,
                derivative_order=order,
            )
        except TweedieNumericalRefusal as exc:
            raise ValueError(f"Tweedie numerical evaluation refused: {exc}") from exc
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluated.log_likelihood,
            parameter_independent_carrier=np.zeros(len(response), dtype=np.float64),
            score=evaluated.score,
            hessian_packed=evaluated.hessian_packed,
            valid=evaluated.valid,
        )

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="TweedieLSS",
        )
        return readonly(parameters[:, 0])

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """``P(Y <= y)`` per row at unit prior weight."""
        values = self._distribution_parameters(theta)
        return readonly(self._distribution_function(y, values, np.ones(len(values))))

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """The ``p``-quantile per row at unit prior weight, ``p`` inside ``(0, 1)``."""
        values = self._distribution_parameters(theta)
        return readonly(self._quantile_function(p, values, np.ones(len(values))))

    def cdf_left_limit(
        self, y: NDArray, theta: NDArray, weights: NDArray | None = None
    ) -> NDArray[np.float64]:
        """``P(Y < y)`` per row: zero on the atom at zero, ``P(Y <= y)`` above it.

        The atom interval a randomised PIT samples on a zero row is therefore
        ``[0, P(Y = 0)]``; the law is continuous everywhere else.
        """
        values = self._distribution_parameters(theta)
        resolved = (
            np.ones(len(values)) if weights is None else _prior_weight_vector(weights, len(values))
        )
        return readonly(self._distribution_function(y, values, resolved, left_limit=True))

    def cdf_prior_weighted(
        self, y: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """``P(Y <= y)`` per row with a prior weight entering as the dispersion ``phi / w``."""
        values = self._distribution_parameters(theta)
        resolved = _prior_weight_vector(weights, len(values))
        return readonly(self._distribution_function(y, values, resolved))

    def quantile_prior_weighted(
        self, p: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """The prior-weighted ``p``-quantile per row, ``p`` inside ``(0, 1)``.

        Probabilities at or below the row's zero mass return the atom itself.
        """
        values = self._distribution_parameters(theta)
        resolved = _prior_weight_vector(weights, len(values))
        return readonly(self._quantile_function(p, values, resolved))

    def variance(self, theta: NDArray) -> NDArray[np.float64]:
        """``Var(Y) = phi mu^p`` per row at unit prior weight."""
        values = self._distribution_parameters(theta)
        return readonly(values[:, 1] * values[:, 0] ** values[:, 2])

    def variance_prior_weighted(self, theta: NDArray, weights: NDArray) -> NDArray[np.float64]:
        """``Var(Y) = phi mu^p / w`` per row: the prior weight enters as ``phi / w``."""
        values = self._distribution_parameters(theta)
        resolved = _prior_weight_vector(weights, len(values))
        return readonly(values[:, 1] * values[:, 0] ** values[:, 2] / resolved)

    def _distribution_parameters(self, theta: NDArray) -> NDArray[np.float64]:
        return _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="TweedieLSS",
        )

    def _distribution_function(
        self,
        y: NDArray,
        values: NDArray[np.float64],
        weights: NDArray[np.float64],
        *,
        left_limit: bool = False,
    ) -> NDArray[np.float64]:
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        if not np.all(np.isfinite(response)):
            raise ValueError("TweedieLSS distribution-function arguments must be finite")
        rate, jump_shape, jump_scale = _compound_poisson_parameters(values, weights)
        if left_limit:
            result = np.zeros(len(values), dtype=np.float64)
        else:
            result = np.where(response < 0.0, 0.0, np.exp(-rate))
        positive = np.flatnonzero(response > 0.0)
        if positive.size:
            terms = _poisson_terms(float(np.max(rate[positive])))
            for rows, block in _series_blocks(rate, jump_shape, jump_scale, positive, terms):
                result[rows] = _block_distribution(block, response[rows])
        return result

    def _quantile_function(
        self,
        p: NDArray,
        values: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        probabilities = _validated_probabilities(p, len(values))
        rate, jump_shape, jump_scale = _compound_poisson_parameters(values, weights)
        result = np.zeros(len(values), dtype=np.float64)
        above = np.flatnonzero(probabilities > np.exp(-rate))
        if above.size == 0:
            return result
        terms = _poisson_terms(float(np.max(rate[above])))
        for rows, block in _series_blocks(
            rate, jump_shape, jump_scale, above, terms, derivatives=True
        ):
            result[rows] = _block_quantile(block, probabilities[rows])
        return result

    def _quantile_bisection(
        self, p: NDArray, theta: NDArray, weights: NDArray | None = None
    ) -> NDArray[np.float64]:
        """The log bisection the bracketed Householder iteration replaced.

        Kept as the reference the fast path is measured against: it inverts the
        same distribution function with no derivative and no starting value, and
        refuses on the same arguments.  Nothing in the engine calls it.
        """
        values = self._distribution_parameters(theta)
        resolved = (
            np.ones(len(values)) if weights is None else _prior_weight_vector(weights, len(values))
        )
        probabilities = _validated_probabilities(p, len(values))
        rate, jump_shape, jump_scale = _compound_poisson_parameters(values, resolved)
        result = np.zeros(len(values), dtype=np.float64)
        above = np.flatnonzero(probabilities > np.exp(-rate))
        if above.size == 0:
            return result
        terms = _poisson_terms(float(np.max(rate[above])))
        for rows, block in _series_blocks(rate, jump_shape, jump_scale, above, terms):
            target = probabilities[rows]
            lower = np.full(len(rows), _LOG_QUANTILE_LOWER)
            upper = np.full(len(rows), _LOG_QUANTILE_UPPER)
            for _ in range(_QUANTILE_BISECTIONS):
                middle = 0.5 * (lower + upper)
                beneath = _block_distribution(block, np.exp(middle)) < target
                lower = np.where(beneath, middle, lower)
                upper = np.where(beneath, upper, middle)
            result[rows] = np.exp(0.5 * (lower + upper))
        return result


__all__ = ["BoundedPowerLink", "TweedieLikelihoodPlan", "TweedieLSS"]
