"""Stable normalized NB2 row likelihood in natural mean-size coordinates.

The exact recurrence sums one term per count unit, so one evaluation costs
``sum(counts)`` recurrence cells. ``_MAX_RECURRENCE_CELLS`` refuses an
evaluation above 2e8 cells and names the two standard remedies: aggregate
identical rows with frequency weights, or move exposure into an offset so the
counts stay small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._common import (
    WeightSemantics,
    readonly,
    validated_derivative_order,
)

_FLOAT = np.float64
_MAX_EXACT_INTEGER = 2**53
_MAX_RECURRENCE_CELLS = 200_000_000
_RECURRENCE_BLOCK_SIZE = 256
_MAX_RECURRENCE_TILE_CELLS = 65_536
_EPS = np.finfo(_FLOAT).eps
_TINY = np.nextafter(_FLOAT(0.0), _FLOAT(1.0))
_DOMAIN_MIN = math.ldexp(1.0, -450)
_DOMAIN_MAX = math.ldexp(1.0, 450)
_MIN_RATIO = math.ldexp(1.0, -26)
_POISSON_LIKE_INITIAL_THETA_FACTOR = 0.5 / math.sqrt(_EPS)
_SERIES_ORDER = 12
_SERIES_RATIO = 1.0 / 16.0
_POWER_SUM_BERNOULLI = (
    1.0,
    -0.5,
    1.0 / 6.0,
    0.0,
    -1.0 / 30.0,
    0.0,
    1.0 / 42.0,
    0.0,
    -1.0 / 30.0,
    0.0,
    5.0 / 66.0,
    0.0,
)


class NegativeBinomialNumericalDomainError(FloatingPointError):
    """A finite NB2 row lies outside the documented binary64 domain."""


class NegativeBinomialInitializationError(ValueError):
    """Raised when NB2 initialization cannot produce an executable state."""


class NegativeBinomialPoissonBoundaryError(NegativeBinomialNumericalDomainError):
    """A row exceeds the finite-θ domain or a fit has a hedged Poisson-like diagnostic."""


class NegativeBinomialDerivativeRepresentationError(NegativeBinomialNumericalDomainError):
    """Natural derivatives cannot retain the direct log-link channels."""


@overload
def _gamma(operations: int) -> float: ...


@overload
def _gamma(operations: NDArray[np.int64]) -> NDArray[np.float64]: ...


def _gamma(operations: int | NDArray[np.int64]) -> float | NDArray[np.float64]:
    scaled = np.asarray(operations, dtype=_FLOAT) * _EPS
    result = scaled / (1.0 - scaled)
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class NegativeBinomialRowEvaluation:
    """Optimizing values and requested signed natural derivatives."""

    optimizing_log_likelihood: NDArray[np.float64]
    score: NDArray[np.float64] | None
    hessian_packed: NDArray[np.float64] | None
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        value = np.asarray(self.optimizing_log_likelihood)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("optimizing_log_likelihood must be a finite one-dimensional array")
        n_rows = len(value)
        object.__setattr__(
            self,
            "optimizing_log_likelihood",
            readonly(value, dtype=np.dtype(np.float64), shape=(n_rows,)),
        )
        object.__setattr__(
            self,
            "valid",
            readonly(self.valid, dtype=np.dtype(np.bool_), shape=(n_rows,)),
        )
        for name, width in (("score", 2), ("hessian_packed", 3)):
            values = getattr(self, name)
            if values is None:
                continue
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(
                self,
                name,
                readonly(values, dtype=np.dtype(np.float64), shape=(n_rows, width)),
            )


def _numeric_vector(values: NDArray, name: str) -> NDArray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if result.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have a real numeric NumPy dtype")
    return result


def _integers(values: NDArray, name: str, *, positive: bool) -> NDArray[np.int64]:
    source = _numeric_vector(values, name)
    if source.dtype.kind == "f":
        invalid = (
            ~np.isfinite(source) | (source != np.floor(source)) | (source > _MAX_EXACT_INTEGER)
        )
        if np.any(invalid):
            raise ValueError(f"{name} must contain exact integers in the float64 integer range")
    elif np.any(source > _MAX_EXACT_INTEGER):
        raise ValueError(f"{name} must be within the exact float64 integer range")
    if np.any(source < (1 if positive else 0)):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must contain {qualifier} exact integers")
    return np.asarray(source, dtype=np.int64)


def recurrence_cells(counts: NDArray) -> int:
    """Return the number of recurrence cells an evaluation of these counts costs (sum of counts).

    The float64 sum is exact for every total the budget can accept and never wraps.
    """
    return int(np.sum(_integers(counts, "counts", positive=False), dtype=np.float64))


def _inputs(
    counts: NDArray,
    mean: NDArray,
    theta: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    derivative_order: int,
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.int64], Literal[0, 1, 2]]:
    sources = tuple(
        _numeric_vector(values, name)
        for values, name in (
            (counts, "counts"),
            (mean, "mean"),
            (theta, "theta"),
            (weights, "weights"),
        )
    )
    if not sources[0].size:
        raise ValueError("negative-binomial row arrays must be non-empty")
    if any(values.shape != sources[0].shape for values in sources[1:]):
        raise ValueError("negative-binomial row arrays must have the same shape")
    integer_counts = _integers(sources[0], "counts", positive=False)
    cells = recurrence_cells(integer_counts)
    if cells > _MAX_RECURRENCE_CELLS:
        raise ValueError(
            f"the exact NB2 recurrence needs {cells:.3g} recurrence cells for this data, above "
            f"the {_MAX_RECURRENCE_CELLS:.0e} budget; aggregate identical "
            "rows with frequency weights, or move exposure into an offset so counts stay small"
        )
    continuous = [np.asarray(values, dtype=_FLOAT) for values in sources[1:]]
    for values, name in zip(continuous, ("mean", "theta", "weights"), strict=True):
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} must be finite and strictly positive")
    if semantics not in ("prior", "frequency"):
        raise ValueError("semantics must be 'prior' or 'frequency'")
    if semantics == "frequency":
        continuous[2] = np.asarray(
            _integers(sources[3], "frequency weights", positive=True),
            dtype=_FLOAT,
        )
    arrays = (np.asarray(integer_counts, dtype=_FLOAT), *continuous)
    return arrays, integer_counts, validated_derivative_order(derivative_order)


def _primitive_initialization_rows(
    response: NDArray,
    exact_count: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> tuple[
    NDArray[np.float64],
    NDArray,
    NDArray[np.float64],
    WeightSemantics,
]:
    if (
        not isinstance(response, np.ndarray)
        or response.dtype != np.dtype(np.float64)
        or response.ndim != 1
        or len(response) == 0
    ):
        raise ValueError("response must be a non-empty one-dimensional float64 NumPy array")
    if (
        not isinstance(exact_count, np.ndarray)
        or exact_count.ndim != 1
        or len(exact_count) == 0
        or (exact_count.dtype != np.dtype(np.float64) and exact_count.dtype.kind not in "iu")
    ):
        raise ValueError(
            "exact_count must be a non-empty one-dimensional float64 or integer NumPy array"
        )
    if (
        not isinstance(weights, np.ndarray)
        or weights.dtype != np.dtype(np.float64)
        or weights.ndim != 1
        or len(weights) == 0
    ):
        raise ValueError("weights must be a non-empty one-dimensional float64 NumPy array")
    if exact_count.shape != response.shape or weights.shape != response.shape:
        raise ValueError("negative-binomial primitive row arrays must have the same shape")
    if np.any(~np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("response must be finite and non-negative")
    if (
        np.any(~np.isfinite(exact_count))
        or np.any(exact_count < 0)
        or np.any(exact_count >= _MAX_EXACT_INTEGER)
    ):
        raise ValueError("exact_count must be in [0, 2**53)")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    if semantics not in ("prior", "frequency"):
        raise ValueError("semantics must be 'prior' or 'frequency'")
    return response, exact_count, weights, semantics


def _require_numerical_domain_range(
    values: NDArray[np.float64],
    name: str,
) -> None:
    outside = (values < _DOMAIN_MIN) | (values > _DOMAIN_MAX)
    if np.any(outside):
        row = int(np.flatnonzero(outside)[0])
        raise NegativeBinomialNumericalDomainError(
            f"negative-binomial {name} at row {row} is outside the numerical domain [2^-450, 2^450]"
        )


def _domain(
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    weights: NDArray[np.float64],
    semantics: WeightSemantics,
) -> tuple[NDArray[np.float64], ...]:
    _require_numerical_domain_range(mean, "mean")
    _require_numerical_domain_range(theta, "theta")
    _require_numerical_domain_range(weights, "weight")
    ones = np.ones_like(weights)
    scale = weights if semantics == "prior" else ones
    multiplier = ones if semantics == "prior" else weights
    effective_mean, effective_theta = scale * mean, scale * theta
    _require_numerical_domain_range(effective_mean, "effective mean")
    _require_numerical_domain_range(effective_theta, "effective theta")
    ratio_bad = np.minimum(effective_mean, effective_theta) < _MIN_RATIO * np.maximum(
        effective_mean, effective_theta
    )
    if np.any(ratio_bad):
        row = int(np.flatnonzero(ratio_bad)[0])
        if effective_theta[row] > effective_mean[row]:
            raise NegativeBinomialPoissonBoundaryError(
                f"negative-binomial row {row} is Poisson-like beyond the supported finite "
                "theta numerical domain"
            )
        raise NegativeBinomialNumericalDomainError(
            f"negative-binomial row {row} has mean/theta ratio outside the numerical domain"
        )
    return scale, multiplier, effective_mean, effective_theta, effective_mean + effective_theta


def _moment_theta_candidate(
    response: NDArray[np.float64],
    weights: NDArray[np.float64],
    semantics: WeightSemantics,
    mean: float,
) -> float | None:
    try:
        mean_square = mean * mean
        residual = response - mean
        if semantics == "frequency":
            numerator = math.fsum(float(weight * mean_square) for weight in weights)
            terms = (weight * (value * value - mean) for weight, value in zip(weights, residual))
        else:
            numerator = math.fsum(mean_square for _ in response)
            terms = (weight * value * value - mean for weight, value in zip(weights, residual))
        denominator = math.fsum(float(value) for value in terms)
        candidate = numerator / denominator
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        return None
    return candidate if math.isfinite(candidate) and candidate > 0.0 else None


def _initial_mean(
    exact_count: NDArray,
    weights: NDArray[np.float64],
    semantics: WeightSemantics,
) -> float:
    weighted_counts = exact_count if semantics == "prior" else weights * exact_count
    try:
        mean = math.fsum(float(value) for value in weighted_counts) / float(np.sum(weights))
    except (OverflowError, ZeroDivisionError) as exc:
        raise NegativeBinomialInitializationError("NB2 initial mean is not representable") from exc
    if not math.isfinite(mean) or mean <= 0.0:
        raise NegativeBinomialInitializationError("NB2 initial mean is not finite and interior")
    return mean


def _resolved_negative_sum(
    terms: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> bool:
    if terms.ndim != 1 or terms.shape != scales.shape or len(terms) == 0:
        return False
    if not np.all(np.isfinite(terms)) or not np.all(np.isfinite(scales)):
        return False
    try:
        total = math.fsum(float(value) for value in terms)
        scale = math.fsum(float(value) for value in scales)
    except (OverflowError, ValueError):
        return False
    operation_error = (32 * len(terms) + 32) * _EPS
    if operation_error >= 1.0:
        return False
    roundoff_bound = operation_error / (1.0 - operation_error) * scale
    return bool(math.isfinite(total) and math.isfinite(roundoff_bound) and total < -roundoff_bound)


def has_resolved_poisson_boundary(
    response: NDArray,
    exact_count: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> bool:
    """Return whether both NB2 Poisson-face directions are resolved as negative."""
    response_values, count_values, weight_values, weight_semantics = _primitive_initialization_rows(
        response, exact_count, weights, semantics
    )
    mean = _initial_mean(count_values, weight_values, weight_semantics)
    with np.errstate(over="ignore", invalid="ignore"):
        residual_square = (response_values - mean) ** 2
        square_scale = (np.abs(response_values) + abs(mean)) ** 2
        if weight_semantics == "frequency":
            moment_terms = weight_values * (residual_square - mean)
            moment_scales = weight_values * (square_scale + abs(mean))
            boundary_terms = weight_values * (residual_square - mean)
            boundary_scales = weight_values * (square_scale + abs(mean))
        else:
            moment_terms = weight_values * residual_square - mean
            moment_scales = weight_values * square_scale + abs(mean)
            boundary_terms = weight_values * residual_square - response_values
            boundary_scales = weight_values * square_scale + np.abs(response_values)
    return bool(
        _resolved_negative_sum(moment_terms, moment_scales)
        and _resolved_negative_sum(
            boundary_terms,
            boundary_scales,
        )
    )


def initialize_negative_binomial(
    response: NDArray,
    exact_count: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    response_values, count_values, weight_values, weight_semantics = _primitive_initialization_rows(
        response, exact_count, weights, semantics
    )
    if np.all(response_values == 0.0):
        raise NegativeBinomialInitializationError("all-zero NB2 samples have no mean initializer")
    mean = _initial_mean(count_values, weight_values, weight_semantics)
    moment = _moment_theta_candidate(response_values, weight_values, weight_semantics, mean)
    poisson_like = mean * _POISSON_LIKE_INITIAL_THETA_FACTOR
    last_error: ValueError | None = None
    for candidate in (poisson_like,) if moment is None else (moment, poisson_like):
        theta = np.column_stack(
            (np.full(len(response_values), mean), np.full(len(response_values), candidate))
        )
        try:
            evaluate_negative_binomial_rows(
                count_values,
                theta[:, 0],
                theta[:, 1],
                weight_values,
                weight_semantics,
                derivative_order=2,
            )
        except ValueError as exc:
            last_error = exc
            continue
        return readonly(theta, dtype=np.dtype(np.float64), shape=(len(response_values), 2))
    raise NegativeBinomialInitializationError(
        "NB2 initialization exhausted its candidates"
    ) from last_error


def _ratio_remainder(
    ratio: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return x/(1+x)-log1p(x) by the lower-error direct or Horner route."""

    rational = np.where(ratio <= 1.0, ratio / (1.0 + ratio), 1.0 - 1.0 / (1.0 + ratio))
    logarithm = np.log1p(ratio)
    result = rational - logarithm
    scale = np.abs(rational) + np.abs(logarithm)
    error = _gamma(4) * scale + _TINY
    candidates = ratio < 1.0
    if not np.any(candidates):
        return result, error, scale

    x = ratio[candidates]
    coefficients = tuple(
        (-1.0) ** (order + 1) * (order - 1.0) / order for order in range(2, _SERIES_ORDER + 1)
    )
    polynomial = np.full_like(x, coefficients[-1])
    absolute = np.full_like(x, abs(coefficients[-1]))
    for coefficient in reversed(coefficients[:-1]):
        polynomial = coefficient + x * polynomial
        absolute = abs(coefficient) + x * absolute
    series = x * x * polynomial
    series_scale = x * x * absolute
    tail = _SERIES_ORDER / (_SERIES_ORDER + 1.0) * x ** (_SERIES_ORDER + 1)
    series_error = _gamma(2 * _SERIES_ORDER + 2) * series_scale + tail + _TINY
    indices = np.flatnonzero(candidates)
    use = series_error < error[candidates]
    chosen = indices[use]
    result[chosen], scale[chosen], error[chosen] = (
        series[use],
        series_scale[use],
        series_error[use],
    )
    return result, error, scale


def _scaled_power_sums(count: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized float64 power sums and absolute roundoff bounds."""

    values = np.zeros(_SERIES_ORDER, dtype=_FLOAT)
    errors = np.zeros(_SERIES_ORDER, dtype=_FLOAT)
    if count < 2:
        values[0] = float(count)
        return values, errors

    inverse_count = _FLOAT(1.0) / _FLOAT(count)
    coefficients = np.zeros(_SERIES_ORDER, dtype=_FLOAT)
    for power in range(_SERIES_ORDER):
        choose = _FLOAT(1.0)
        for term in range(power + 1):
            coefficients[term] = choose * _POWER_SUM_BERNOULLI[term]
            if term < power:
                choose *= _FLOAT(power + 1 - term) / _FLOAT(term + 1)
        polynomial = coefficients[power]
        absolute = abs(polynomial)
        for term in range(power - 1, -1, -1):
            polynomial = coefficients[term] + inverse_count * polynomial
            absolute = abs(coefficients[term]) + inverse_count * absolute
        denominator = _FLOAT(power + 1)
        values[power] = polynomial / denominator
        errors[power] = _gamma(4 * power + 8) * absolute / denominator + _TINY
    return values, errors


def _series_hessian_step(
    order: int,
    coefficient: float,
    coefficient_abs: float,
    coefficient_error: float,
    x: float,
    state: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Advance only the Hessian Horner state for one series coefficient."""

    weighted, transformed, weighted_absolute, weighted_error = state
    return (
        order * coefficient + x * weighted,
        (1.0 - order) * coefficient + x * transformed,
        order * coefficient_abs + x * weighted_absolute,
        order * coefficient_error + x * weighted_error,
    )


def _theta_series(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    derivative_order: Literal[1, 2],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the fixed order-12 series only when its error beats recurrence."""

    n_rows = len(counts)
    use = np.zeros((n_rows, derivative_order), dtype=np.bool_)
    natural = np.zeros((n_rows, derivative_order), dtype=_FLOAT)
    direct = np.zeros((n_rows, derivative_order), dtype=_FLOAT)
    candidate_scale = np.maximum(mean, np.maximum(counts.astype(_FLOAT) - 1.0, 0.0))
    candidates = np.flatnonzero(candidate_scale / theta <= _SERIES_RATIO)

    for row in candidates:
        count, row_mean, row_theta = int(counts[row]), float(mean[row]), float(theta[row])
        scale = float(candidate_scale[row])
        x = scale / row_theta
        polynomial = absolute = coefficient_error_absolute = 0.0
        if derivative_order == 2:
            weighted = transformed = weighted_absolute = weighted_error = 0.0
        sums, sum_errors = _scaled_power_sums(count)
        count_scale = _FLOAT(count) / _FLOAT(scale) if count > 1 else _FLOAT(0.0)
        for order in range(_SERIES_ORDER, 1, -1):
            if count > 1:
                offset_factor = float(count_scale**order)
                offset = float(sums[order - 1]) * offset_factor
                offset_error = (
                    abs(offset_factor) * float(sum_errors[order - 1])
                    + _gamma(order + 2) * abs(offset)
                    + _TINY
                )
            else:
                offset = offset_error = 0.0
            mean_term = (row_mean / scale) ** (order - 1) * (
                ((order - 1.0) / order * row_mean - count) / scale
            )
            sign = -1.0 if order % 2 == 0 else 1.0
            coefficient = sign * (offset + mean_term)
            coefficient_abs = abs(offset) + abs(mean_term)
            polynomial = coefficient + x * polynomial
            absolute = coefficient_abs + x * absolute
            coefficient_error_absolute = offset_error + x * coefficient_error_absolute
            if derivative_order == 2:
                weighted, transformed, weighted_absolute, weighted_error = _series_hessian_step(
                    order,
                    coefficient,
                    coefficient_abs,
                    offset_error,
                    x,
                    (weighted, transformed, weighted_absolute, weighted_error),
                )

        natural[row, 0] = x * x * polynomial
        direct[row, 0] = scale * x * polynomial
        mean_ratio = row_mean / row_theta
        offset_ratio = max(count - 1, 0) / row_theta
        delta = abs(row_mean - count)
        score_tail = (
            count / row_theta * offset_ratio**_SERIES_ORDER / (1.0 - offset_ratio)
            + mean_ratio ** (_SERIES_ORDER + 1) / (_SERIES_ORDER + 1.0)
            + delta / row_theta * mean_ratio**_SERIES_ORDER / (1.0 - mean_ratio)
        )
        series_score_error = (
            x * x * (_gamma(80) * absolute + (1.0 + _gamma(32)) * coefficient_error_absolute)
            + score_tail
            + _TINY
        )
        total = row_mean + row_theta
        recurrence_score_scale = count / total * (
            (row_mean + max(count - 1, 0)) / row_theta
        ) + float(remainder_scale[row])
        recurrence_score_error = _gamma(count + 64) * recurrence_score_scale + float(
            remainder_error[row]
        )
        use[row, 0] = series_score_error < recurrence_score_error

        if derivative_order == 1:
            continue
        natural[row, 1] = -(x * x / row_theta) * weighted
        direct[row, 1] = scale * x * transformed
        hessian_tail = (
            count
            / row_theta**2
            * offset_ratio**_SERIES_ORDER
            * (_SERIES_ORDER + 1.0 - _SERIES_ORDER * offset_ratio)
            / (1.0 - offset_ratio) ** 2
            + mean_ratio ** (_SERIES_ORDER + 1) / row_theta / (1.0 - mean_ratio)
            + delta
            / row_theta**2
            * mean_ratio**_SERIES_ORDER
            * (_SERIES_ORDER + 1.0 - _SERIES_ORDER * mean_ratio)
            / (1.0 - mean_ratio) ** 2
        )
        series_hessian_error = (
            x
            * x
            / row_theta
            * (_gamma(96) * weighted_absolute + (1.0 + _gamma(32)) * weighted_error)
            + hessian_tail
            + _TINY
        )
        recurrence_hessian_scale = (
            recurrence_score_scale * (1.0 / row_theta + 1.0 / total)
            + (row_mean / total) * mean_ratio / total
        )
        use[row, 1] = series_hessian_error < (_gamma(count + 80) * recurrence_hessian_scale + _TINY)

    return use, natural, direct


def _iter_recurrence_tiles(
    counts: NDArray[np.int64],
    *,
    block_size: int = _RECURRENCE_BLOCK_SIZE,
    max_cells: int = _MAX_RECURRENCE_TILE_CELLS,
):
    """Yield exact-width tiles whose two-dimensional size is bounded."""

    if not 1 <= block_size <= max_cells:
        raise ValueError("recurrence tile limits require 1 <= block_size <= max_cells")
    positive_rows = np.flatnonzero(counts > 0)
    if not positive_rows.size:
        return
    order = np.argsort(counts[positive_rows], kind="heapsort")
    rows, sorted_counts = positive_rows[order], counts[positive_rows[order]]
    for start in range(0, int(sorted_counts[-1]), block_size):
        stop = min(start + block_size, int(sorted_counts[-1]))
        active_start = int(np.searchsorted(sorted_counts, start, side="right"))
        full_start = int(np.searchsorted(sorted_counts, stop, side="left"))
        group_start = active_start
        while group_start < full_start:
            count = int(sorted_counts[group_start])
            group_stop = min(int(np.searchsorted(sorted_counts, count, side="right")), full_start)
            width = count - start
            capacity = max_cells // width
            for tile_start in range(group_start, group_stop, capacity):
                yield rows[tile_start : min(tile_start + capacity, group_stop)], start, width
            group_start = group_stop
        width = stop - start
        capacity = max_cells // width
        for tile_start in range(full_start, len(rows), capacity):
            yield rows[tile_start : min(tile_start + capacity, len(rows))], start, width


def _recurrences(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    derivative_order: Literal[0, 1, 2],
    derivative_rows: NDArray[np.bool_],
) -> tuple:
    """Accumulate bounded value, natural-theta, and direct log-theta terms."""

    n_rows = len(counts)
    value = np.zeros(n_rows, dtype=_FLOAT)
    natural = np.zeros((n_rows, derivative_order), dtype=_FLOAT)
    direct = np.zeros((n_rows, derivative_order), dtype=_FLOAT)

    for rows, start, width in _iter_recurrence_tiles(counts):
        assert rows.size * width <= _MAX_RECURRENCE_TILE_CELLS
        offsets = np.arange(start, start + width, dtype=_FLOAT)[None, :]
        row_mean, row_theta = mean[rows, None], theta[rows, None]
        total = row_mean + row_theta
        value[rows] += np.sum(np.log1p((offsets - row_mean) / total), axis=1)
        selected = rows[derivative_rows[rows]]
        if derivative_order == 0 or not selected.size:
            continue
        row_mean, row_theta = mean[selected, None], theta[selected, None]
        total = row_mean + row_theta
        inverse_offset, inverse_total = 1.0 / (row_theta + offsets), 1.0 / total
        score_terms = (row_mean - offsets) * inverse_offset * inverse_total
        direct_terms = (row_theta * inverse_offset) * ((row_mean - offsets) / total)
        natural[selected, 0] += np.sum(score_terms, axis=1)
        direct[selected, 0] += np.sum(direct_terms, axis=1)
        if derivative_order == 1:
            continue
        hessian_terms = -score_terms * (inverse_offset + inverse_total)
        direct_hessian_terms = direct_terms * (
            1.0 - row_theta * inverse_offset - row_theta * inverse_total
        )
        natural[selected, 1] += np.sum(hessian_terms, axis=1)
        direct[selected, 1] += np.sum(direct_hessian_terms, axis=1)
    return value, natural, direct


def _retain_log_channels(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    score: NDArray[np.float64],
    hessian: NDArray[np.float64] | None,
    target_score: NDArray[np.float64],
    target_hessian: NDArray[np.float64] | None,
    score_scale: NDArray[np.float64],
    hessian_scale: NDArray[np.float64] | None,
) -> None:
    """Refuse natural channels that discard independently computed log channels."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        reconstructed_score = score * np.column_stack((mean, theta))
    bound = (
        _gamma(counts + 128)[:, None]
        * (score_scale + np.abs(target_score) + np.abs(reconstructed_score))
        + 16.0 * _TINY
    )
    retained = np.all(
        np.isfinite(reconstructed_score) & (np.abs(reconstructed_score - target_score) <= bound),
        axis=1,
    )
    if hessian is not None:
        assert target_hessian is not None and hessian_scale is not None
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            reconstructed_hessian = np.column_stack(
                (
                    (hessian[:, 0] * mean) * mean + score[:, 0] * mean,
                    (hessian[:, 1] * mean) * theta,
                    (hessian[:, 2] * theta) * theta + score[:, 1] * theta,
                )
            )
        bound = (
            _gamma(counts + 160)[:, None]
            * (hessian_scale + np.abs(target_hessian) + np.abs(reconstructed_hessian))
            + 24.0 * _TINY
        )
        retained &= np.all(
            np.isfinite(reconstructed_hessian)
            & (np.abs(reconstructed_hessian - target_hessian) <= bound),
            axis=1,
        )
    if not np.all(retained):
        row = int(np.flatnonzero(~retained)[0])
        raise NegativeBinomialDerivativeRepresentationError(
            f"negative-binomial row {row} cannot retain its direct log-link channels "
            "through float64 natural derivatives within the numerical domain"
        )


def _finite_output(
    values: NDArray[np.float64],
    name: str,
    error: type[NegativeBinomialNumericalDomainError],
) -> None:
    finite_rows = np.all(np.isfinite(values.reshape(len(values), -1)), axis=1)
    if not np.all(finite_rows):
        row = int(np.flatnonzero(~finite_rows)[0])
        raise error(f"negative-binomial {name} at row {row} is not finite in the numerical domain")


def evaluate_negative_binomial_rows(
    counts: NDArray,
    mean: NDArray,
    theta: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    *,
    derivative_order: int = 2,
) -> NegativeBinomialRowEvaluation:
    """Evaluate normalized NB2 rows, excluding the factorial carrier.

    Continuous inputs and effective prior-scaled parameters must be in
    ``[2^-450, 2^450]`` with effective ``min(mean, theta) / max(mean, theta)``
    at least ``2^-26``. Hessians are packed as ``(mean-mean, mean-theta,
    theta-theta)`` in natural coordinates.
    """

    arrays, integer_counts, order = _inputs(
        counts, mean, theta, weights, semantics, derivative_order
    )
    count_values, mean_values, theta_values, weight_values = arrays
    scale, multiplier, effective_mean, effective_theta, total = _domain(
        mean_values, theta_values, weight_values, semantics
    )

    if order:
        remainder, remainder_error, remainder_scale = _ratio_remainder(
            effective_mean / effective_theta
        )
        series_use, series_natural, series_direct = _theta_series(
            integer_counts,
            effective_mean,
            effective_theta,
            remainder_error,
            remainder_scale,
            order,
        )
        derivative_rows = ~np.all(series_use, axis=1)
    else:
        derivative_rows = np.zeros(len(integer_counts), dtype=np.bool_)

    value_recurrence, recurrence_natural, recurrence_direct = _recurrences(
        integer_counts,
        effective_mean,
        effective_theta,
        order,
        derivative_rows,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        effective_value = (
            value_recurrence
            + count_values * np.log(effective_mean)
            - effective_theta * np.log1p(effective_mean / effective_theta)
        )
        optimizing = multiplier * effective_value
    _finite_output(optimizing, "value", NegativeBinomialNumericalDomainError)

    scores = hessians = None
    if order:
        recurrence_score = recurrence_natural[:, 0] + remainder
        recurrence_direct_score = recurrence_direct[:, 0] + effective_theta * remainder
        theta_score = np.where(series_use[:, 0], series_natural[:, 0], recurrence_score)
        direct_theta_score = np.where(
            series_use[:, 0], series_direct[:, 0], recurrence_direct_score
        )
        recurrence_term_scale = (
            count_values
            / total
            * ((effective_mean + np.maximum(count_values - 1.0, 0.0)) / effective_theta)
        )
        direct_theta_score_scale = effective_theta * (recurrence_term_scale + remainder_scale)
        difference = count_values - effective_mean
        mean_score = (effective_theta / total) * (difference / effective_mean)
        direct_mean_score = (effective_theta / total) * difference
        factor = multiplier * scale
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scores = factor[:, None] * np.column_stack((mean_score, theta_score))
            target_score = multiplier[:, None] * np.column_stack(
                (direct_mean_score, direct_theta_score)
            )
        mean_score_scale = (effective_theta / total) * (np.abs(count_values) + effective_mean)
        target_score_scale = multiplier[:, None] * np.column_stack(
            (mean_score_scale, direct_theta_score_scale)
        )
        _finite_output(scores, "score", NegativeBinomialDerivativeRepresentationError)

        target_hessian = target_hessian_scale = None
        if order == 2:
            hessian_remainder = (
                (effective_mean / total) * (effective_mean / effective_theta) / total
            )
            direct_hessian_remainder = (
                effective_theta * (effective_mean / total) ** 2 + effective_theta * remainder
            )
            recurrence_hessian = recurrence_natural[:, 1] + hessian_remainder
            recurrence_direct_hessian = recurrence_direct[:, 1] + direct_hessian_remainder
            theta_hessian = np.where(series_use[:, 1], series_natural[:, 1], recurrence_hessian)
            direct_theta_hessian = np.where(
                series_use[:, 1], series_direct[:, 1], recurrence_direct_hessian
            )
            direct_theta_hessian_scale = (
                effective_theta
                * recurrence_term_scale
                * (1.0 + effective_theta * (1.0 / effective_theta + 1.0 / total))
                + effective_theta * (effective_mean / total) ** 2
                + effective_theta * remainder_scale
            )
            count_over_mean = count_values / effective_mean
            mean_hessian = -(effective_theta / total) * (
                count_over_mean / effective_mean + (count_over_mean - 1.0) / total
            )
            cross_hessian = difference / (total * total)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                hessians = (factor * scale)[:, None] * np.column_stack(
                    (mean_hessian, cross_hessian, theta_hessian)
                )
                direct_mean_hessian = -(
                    effective_mean
                    / total
                    * (effective_theta / total)
                    * (count_values + effective_theta)
                )
                direct_cross_hessian = (
                    effective_mean / total * (effective_theta / total) * difference
                )
                target_hessian = multiplier[:, None] * np.column_stack(
                    (
                        direct_mean_hessian,
                        direct_cross_hessian,
                        direct_theta_hessian,
                    )
                )
            target_hessian_scale = multiplier[:, None] * np.column_stack(
                (
                    np.abs(direct_mean_hessian) + mean_score_scale,
                    np.abs(direct_cross_hessian),
                    direct_theta_hessian_scale + direct_theta_score_scale,
                )
            )
            _finite_output(hessians, "Hessian", NegativeBinomialDerivativeRepresentationError)

        _retain_log_channels(
            integer_counts,
            mean_values,
            theta_values,
            scores,
            hessians,
            target_score,
            target_hessian,
            target_score_scale,
            target_hessian_scale,
        )

    return NegativeBinomialRowEvaluation(
        optimizing_log_likelihood=optimizing,
        score=scores,
        hessian_packed=hessians,
        valid=np.ones(len(integer_counts), dtype=np.bool_),
    )


__all__ = [
    "NegativeBinomialDerivativeRepresentationError",
    "NegativeBinomialInitializationError",
    "NegativeBinomialNumericalDomainError",
    "NegativeBinomialPoissonBoundaryError",
    "NegativeBinomialRowEvaluation",
    "evaluate_negative_binomial_rows",
    "has_resolved_poisson_boundary",
    "initialize_negative_binomial",
    "recurrence_cells",
]
