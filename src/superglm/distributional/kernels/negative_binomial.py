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
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.kernels._negative_binomial_numba as _compiled
from superglm._parallel import pooled_row_blocks
from superglm._roundoff import EPS as _EPS
from superglm._roundoff import SMALLEST_SUBNORMAL as _SUBNORMAL
from superglm._roundoff import gamma
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
_DOMAIN_MIN = math.ldexp(1.0, -450)
_DOMAIN_MAX = math.ldexp(1.0, 450)
# The kernel is NB2 at every mean/theta ratio; no ratio is refused.
#
# Toward the Poisson limit (theta -> inf) NB2 is the Poisson law plus
# ((y - mean)^2 - y)/(2 theta) + O(theta^-2) (the NB2 expansion in 1/theta).
# Once max(mean, count - 1) < _POISSON_LIMIT_RATIO * theta, in effective
# units, that correction, and with it every derivative in theta, is below one
# unit roundoff of the row's own value and mean channels: the row has reached
# its Poisson limit at working precision. It is evaluated there, at the
# family's boundary theta = inf, as a likelihood whose maximum lies at
# infinity is read in the limit model (Geyer, Electron. J. Statist. 3, 2009):
# at one surrogate size, 2^_POISSON_SURROGATE_EXPONENT times that reach,
# whatever theta is, with every theta channel exactly zero. Its value and mean
# channels then agree with the Poisson row's to about 2^-60 relative, they are
# flat in theta, and a size that has reached the limit is not pushed further
# by derivatives below the precision of the fit. The ratio bound also keeps
# every other row's natural theta channels in the normal range: their extreme
# exponent, (weight/theta) (mean/theta)^2 at the +/-450 limits, is -1004.
#
# Toward the other limit (theta/mean -> 0) the value recurrence switches from
# log1p((j - mean)/(mean + theta)), which approaches log1p(-1), to
# log((theta + j)/(mean + theta)) once that argument is below -1/2.
_POISSON_LIMIT_RATIO = _EPS
_POISSON_SURROGATE_EXPONENT = 60
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
_SERIES_BERNOULLI = np.array(_POWER_SUM_BERNOULLI, dtype=_FLOAT)
# Exponents as data: the compiled series calls libm pow exactly where the
# reference loop did (see _negative_binomial_numba).
_SERIES_EXPONENTS = np.arange(_SERIES_ORDER + 2, dtype=_FLOAT)
_PARALLEL_MIN_ROWS = 5_000


class NegativeBinomialNumericalDomainError(FloatingPointError):
    """A finite NB2 row lies outside the documented binary64 domain."""


class NegativeBinomialInitializationError(ValueError):
    """Raised when NB2 initialization cannot produce an executable state."""


class NegativeBinomialPoissonBoundaryError(NegativeBinomialNumericalDomainError):
    """A row exceeds the finite-θ domain or a fit has a hedged Poisson-like diagnostic."""


class NegativeBinomialDerivativeRepresentationError(NegativeBinomialNumericalDomainError):
    """Natural derivatives cannot retain the direct log-link channels."""


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
    counts: NDArray[np.int64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """Effective parameters, with Poisson-limit rows moved to their surrogate size.

    Returns ``(scale, multiplier, effective_mean, effective_theta, total,
    poisson)``. A row is at its Poisson limit when ``max(mean, count - 1)``,
    in effective units, is below ``_POISSON_LIMIT_RATIO`` times its size (the
    expected information, which has no count, uses the mean alone). Its
    effective size is then its surrogate, so it is in the numerical domain
    whatever its own size.
    """
    _require_numerical_domain_range(mean, "mean")
    _require_numerical_domain_range(weights, "weight")
    ones = np.ones_like(weights)
    scale = weights if semantics == "prior" else ones
    multiplier = ones if semantics == "prior" else weights
    effective_mean = scale * mean
    _require_numerical_domain_range(effective_mean, "effective mean")
    reach = (
        effective_mean
        if counts is None
        else np.maximum(effective_mean, np.maximum(counts.astype(_FLOAT) - 1.0, 0.0))
    )
    with np.errstate(over="ignore"):
        effective_theta = scale * theta
    poisson = reach < _POISSON_LIMIT_RATIO * effective_theta
    if np.any(poisson):
        surrogate = np.minimum(np.ldexp(reach[poisson], _POISSON_SURROGATE_EXPONENT), _DOMAIN_MAX)
        if np.any(reach[poisson] >= _POISSON_LIMIT_RATIO * surrogate):
            raise NegativeBinomialNumericalDomainError(
                "a negative-binomial row's Poisson limit lies outside the numerical domain"
            )
        effective_theta = effective_theta.copy()
        effective_theta[poisson] = surrogate
    _require_numerical_domain_range(theta[~poisson], "theta")
    _require_numerical_domain_range(effective_theta, "effective theta")
    total = effective_mean + effective_theta
    return scale, multiplier, effective_mean, effective_theta, total, poisson


def _moment_theta_candidate(
    response: NDArray[np.float64],
    weights: NDArray[np.float64],
    semantics: WeightSemantics,
    mean: float,
) -> float | None:
    try:
        mean_square = mean * mean
        residual = response - mean
        # Elementwise float64 terms as before, summed exactly: fsum of a list
        # equals fsum of the same values from a generator.
        if semantics == "frequency":
            numerator = math.fsum((weights * mean_square).tolist())
            terms = weights * (residual * residual - mean)
        else:
            numerator = math.fsum([mean_square] * len(response))
            terms = weights * residual * residual - mean
        denominator = math.fsum(terms.tolist())
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
        mean = math.fsum(np.asarray(weighted_counts, dtype=_FLOAT).tolist()) / float(
            np.sum(weights)
        )
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
        total = math.fsum(terms.tolist())
        scale = math.fsum(scales.tolist())
    except (OverflowError, ValueError):
        return False
    operation_error = (32 * len(terms) + 32) * _EPS
    if operation_error >= 1.0:
        return False
    roundoff_bound = operation_error / (1.0 - operation_error) * scale
    return bool(math.isfinite(total) and math.isfinite(roundoff_bound) and total < -roundoff_bound)


def poisson_limit_row_count(mean, theta, weights, semantics, counts) -> int:
    """Rows past their Poisson limit, by the mask the likelihood kernel itself applies.

    This is where the kernel stops evaluating the size, not where the size stops
    being identified: a size just short of the threshold is as flat in the
    likelihood and is not counted, so the count is a lower bound that depends on
    where the solver stopped as well as on the data.
    """
    return int(np.count_nonzero(_domain(mean, theta, weights, semantics, counts)[-1]))


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
    """Return starting mean and dispersion values for a negative binomial fit."""

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
    error = gamma(8) * scale + _SUBNORMAL
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
    series_error = gamma(2 * (2 * _SERIES_ORDER + 2)) * series_scale + tail + _SUBNORMAL
    indices = np.flatnonzero(candidates)
    use = series_error < error[candidates]
    chosen = indices[use]
    result[chosen], scale[chosen], error[chosen] = (
        series[use],
        series_scale[use],
        series_error[use],
    )
    return result, error, scale


def _theta_series(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    derivative_order: Literal[1, 2],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the fixed order-12 series only when its error beats recurrence.

    Candidate rows run through the compiled row kernel, whose arithmetic is the
    per-row reference's operation for operation (see ``_negative_binomial_numba``).
    """

    n_rows = len(counts)
    use = np.zeros((n_rows, derivative_order), dtype=np.bool_)
    natural = np.zeros((n_rows, derivative_order), dtype=_FLOAT)
    direct = np.zeros((n_rows, derivative_order), dtype=_FLOAT)
    candidate_scale = np.maximum(mean, np.maximum(counts.astype(_FLOAT) - 1.0, 0.0))
    candidates = np.flatnonzero(candidate_scale / theta <= _SERIES_RATIO).astype(np.int64)
    if not candidates.size:
        return use, natural, direct
    operands = (
        np.ascontiguousarray(counts, dtype=np.int64),
        np.ascontiguousarray(mean, dtype=_FLOAT),
        np.ascontiguousarray(theta, dtype=_FLOAT),
        candidate_scale,
        np.ascontiguousarray(remainder_error, dtype=_FLOAT),
        np.ascontiguousarray(remainder_scale, dtype=_FLOAT),
        int(derivative_order),
        _SERIES_ORDER,
        _SERIES_EXPONENTS,
        _SERIES_BERNOULLI,
        use,
        natural,
        direct,
    )
    if candidates.size < _PARALLEL_MIN_ROWS:
        _compiled.theta_series_serial(candidates, *operands)
        return use, natural, direct
    # Each candidate row writes only its own entries: blocks of candidates run
    # on the worker pool and give the serial core's bits at any worker count.
    pooled_row_blocks(
        lambda start, stop: _compiled.theta_series_serial(candidates[start:stop], *operands),
        candidates.size,
        _compiled._PARALLEL_BLOCK,
    )
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
        # log1p((j - mean)/total) == log((theta + j)/total). Below -1/2 the
        # log1p argument nears -1, where its rounding is amplified by
        # 1/(1 + argument); the quotient form is then the well-conditioned one.
        # A tile without such a cell (mean < theta + 2j throughout, the usual
        # claim-frequency case) takes log1p alone. A tile with one evaluates
        # both forms and selects: numpy's masked (where=) and gathered
        # evaluations of the logarithm measured slower than the full one.
        argument = (offsets - row_mean) / total
        quotient = argument < -0.5
        if quotient.any():
            # The discarded log1p of a quotient cell may be log1p(-1) = -inf; only
            # the quotient form is kept there, so its divide warning is expected.
            with np.errstate(divide="ignore"):
                discarded = np.log1p(argument)
            terms = np.where(quotient, np.log((row_theta + offsets) / total), discarded)
        else:
            terms = np.log1p(argument)
        value[rows] += np.sum(terms, axis=1)
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
        gamma(2 * (counts + 128))[:, None]
        * (score_scale + np.abs(target_score) + np.abs(reconstructed_score))
        + 16.0 * _SUBNORMAL
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
            gamma(2 * (counts + 160))[:, None]
            * (hessian_scale + np.abs(target_hessian) + np.abs(reconstructed_hessian))
            + 24.0 * _SUBNORMAL
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
    ``[2^-450, 2^450]``, except the size of a row past its Poisson limit
    (effective ``max(mean, count - 1) < 2^-52 theta``), which may take any
    positive value and has zero theta channels. Hessians are packed as ``(mean-mean,
    mean-theta, theta-theta)`` in natural coordinates.
    """

    arrays, integer_counts, order = _inputs(
        counts, mean, theta, weights, semantics, derivative_order
    )
    count_values, mean_values, theta_values, weight_values = arrays
    scale, multiplier, effective_mean, effective_theta, total, poisson = _domain(
        mean_values, theta_values, weight_values, semantics, integer_counts
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

        if np.any(poisson):
            # A Poisson-limit row has no theta channels (see _POISSON_LIMIT_RATIO).
            for channels in (scores, target_score):
                channels[poisson, 1] = 0.0
            for channels in (hessians, target_hessian):
                if channels is not None:
                    channels[poisson, 1:] = 0.0
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


def _poisson_limit_theta_channels(
    derivatives: NDArray[np.float64],
    error: NDArray[np.float64],
    poisson: NDArray[np.bool_],
) -> None:
    """Zero a Poisson-limit row's log-theta derivatives, in place, keeping a true bound.

    Column 0 is the pure log-mean derivative; every later column involves
    log theta. The last, pure log-theta column is the certified one: its true
    value at any size past the surrogate is smaller in magnitude than at the
    surrogate (it falls like 1/theta there), so the bound grows by that
    magnitude, rounded upward once. The mixed columns are below the row's
    unit roundoff, as its theta score is.
    """

    if not np.any(poisson):
        return
    with np.errstate(over="ignore"):
        error[poisson] = np.nextafter(error[poisson] + np.abs(derivatives[poisson, -1]), np.inf)
    derivatives[poisson, 1:] = 0.0


def negative_binomial_log_third_derivatives(
    counts: NDArray,
    mean: NDArray,
    theta: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Third derivatives of the NB2 row log-likelihood in ``(log mean, log theta)``.

    Returns ``(third, error)``: ``third[:, k]`` is ``l_aaa, l_aab, l_abb, l_bbb``
    with ``a = log mean`` and ``b = log theta``, and ``error`` a certified
    absolute bound on the rounding and truncation error of ``l_bbb``.  The
    first three are closed forms with relative error at most ``gamma_11`` of
    their absolute-term scale.  ``l_bbb`` cancels like the theta score as the
    row approaches Poisson, so it is the near-Poisson series of the theta score
    differentiated twice more in ``log theta`` (coefficients ``(n - 1)^2 c_n``)
    or the count recurrence, whichever certifies the smaller error, as the
    score and Hessian channels choose.  Inputs and domain are the row kernel's.
    """

    arrays, integer_counts, _ = _inputs(counts, mean, theta, weights, semantics, 2)
    _, _, _, weight_values = arrays
    _, multiplier, effective_mean, effective_theta, _, poisson = _domain(
        arrays[1], arrays[2], weight_values, semantics, integer_counts
    )
    remainder, remainder_error, remainder_scale = _ratio_remainder(effective_mean / effective_theta)
    third, error, _ = _compiled.log_third_derivatives(
        np.ascontiguousarray(integer_counts, dtype=np.int64),
        np.ascontiguousarray(effective_mean, dtype=_FLOAT),
        np.ascontiguousarray(effective_theta, dtype=_FLOAT),
        np.ascontiguousarray(remainder, dtype=_FLOAT),
        np.ascontiguousarray(remainder_error, dtype=_FLOAT),
        np.ascontiguousarray(remainder_scale, dtype=_FLOAT),
        _SERIES_RATIO,
        _SERIES_ORDER,
        _SERIES_EXPONENTS,
        _SERIES_BERNOULLI,
    )
    _poisson_limit_theta_channels(third, error, poisson)
    with np.errstate(over="ignore", invalid="ignore"):
        third = third * multiplier[:, None]
        error = (1.0 + gamma(4)) * (error * multiplier)
    _finite_output(third, "third derivative", NegativeBinomialDerivativeRepresentationError)
    _finite_output(error, "third-derivative bound", NegativeBinomialDerivativeRepresentationError)
    return third, error


def negative_binomial_certified_predictor_curvature_directional(
    counts: NDArray,
    eta: NDArray,
    eta_direction: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The packed curvature direction and an entrywise bound on its certified error.

    ``l_bbb`` is the one channel computed only to a certified bound ``e``
    (series truncation and recurrence rounding, which cancel by ``theta^2``
    near Poisson), and it enters only the ``bb`` entry, multiplied by
    ``d_b``; so the certificate is ``|d_b| e`` on ``bb``, rounded upward once
    for the one rounded product, and zero on ``aa`` and ``ab``, whose third
    derivatives are closed forms.
    """

    try:
        predictors = np.asarray(eta, dtype=_FLOAT)
        direction = np.asarray(eta_direction, dtype=_FLOAT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NB2 predictor state and direction must be finite n-by-2 arrays") from exc
    expected = (len(np.asarray(counts)), 2)
    if (
        predictors.shape != expected
        or direction.shape != expected
        or not np.all(np.isfinite(predictors))
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("NB2 predictor state and direction must be finite n-by-2 arrays")
    with np.errstate(over="ignore", under="ignore"):
        mean = np.exp(predictors[:, 0])
        theta = np.exp(predictors[:, 1])
    third, error = negative_binomial_log_third_derivatives(counts, mean, theta, weights, semantics)
    along_mean, along_theta = direction[:, 0], direction[:, 1]
    result = np.empty((len(mean), 3), dtype=_FLOAT)
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, 0] = -(third[:, 0] * along_mean + third[:, 1] * along_theta)
        result[:, 1] = -(third[:, 1] * along_mean + third[:, 2] * along_theta)
        result[:, 2] = -(third[:, 2] * along_mean + third[:, 3] * along_theta)
    _finite_output(result, "curvature direction", NegativeBinomialDerivativeRepresentationError)
    certificate = np.zeros_like(result)
    with np.errstate(over="ignore"):
        certificate[:, 2] = np.nextafter(np.abs(along_theta) * error, np.inf)
    _finite_output(
        certificate, "curvature-direction bound", NegativeBinomialDerivativeRepresentationError
    )
    return (
        readonly(result, dtype=np.dtype(np.float64), shape=result.shape),
        readonly(certificate, dtype=np.dtype(np.float64), shape=certificate.shape),
    )


def negative_binomial_log_fourth_derivatives(
    counts: NDArray,
    mean: NDArray,
    theta: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fourth derivatives of the NB2 row log-likelihood in ``(log mean, log theta)``.

    Returns ``(fourth, error)``: ``fourth[:, k]`` is ``l_aaaa, l_aaab, l_aabb,
    l_abbb, l_bbbb`` and ``error`` a bound on the series truncation and
    recurrence rounding of ``l_bbbb``.  The first four never touch the
    log-gamma difference and are closed forms in ``q = mu/(mu + theta)`` and
    ``p = theta/(mu + theta)`` (generated, homogenised in ``(q, p)``);
    ``l_bbbb`` cancels like ``l_bbb`` near Poisson and takes the series or the
    recurrence as ``l_bbb`` does, one log-theta derivative further.  Inputs
    and domain are the row kernel's.
    """
    from superglm.distributional.kernels._curvature_direction import stack_columns
    from superglm.distributional.kernels._fourth_generated import (
        negative_binomial_log_fourth_closed,
    )

    arrays, integer_counts, _ = _inputs(counts, mean, theta, weights, semantics, 2)
    _, _, _, weight_values = arrays
    _, multiplier, effective_mean, effective_theta, total, poisson = _domain(
        arrays[1], arrays[2], weight_values, semantics, integer_counts
    )
    remainder, remainder_error, remainder_scale = _ratio_remainder(effective_mean / effective_theta)
    bbbb, error = _compiled.log_fourth_derivative(
        np.ascontiguousarray(integer_counts, dtype=np.int64),
        np.ascontiguousarray(effective_mean, dtype=_FLOAT),
        np.ascontiguousarray(effective_theta, dtype=_FLOAT),
        np.ascontiguousarray(remainder, dtype=_FLOAT),
        np.ascontiguousarray(remainder_error, dtype=_FLOAT),
        np.ascontiguousarray(remainder_scale, dtype=_FLOAT),
        _SERIES_RATIO,
        _SERIES_ORDER,
        _SERIES_EXPONENTS,
        _SERIES_BERNOULLI,
    )
    n_rows = len(effective_mean)
    fourth = np.empty((n_rows, 5), dtype=_FLOAT)
    with np.errstate(over="ignore", invalid="ignore"):
        fourth[:, :4] = stack_columns(
            negative_binomial_log_fourth_closed(
                arrays[0], effective_theta, effective_mean / total, effective_theta / total
            ),
            n_rows,
        )
        fourth[:, 4] = bbbb
        _poisson_limit_theta_channels(fourth, error, poisson)
        fourth *= multiplier[:, None]
        error = (1.0 + gamma(4)) * (error * multiplier)
    _finite_output(fourth, "fourth derivative", NegativeBinomialDerivativeRepresentationError)
    _finite_output(error, "fourth-derivative bound", NegativeBinomialDerivativeRepresentationError)
    return fourth, error


def negative_binomial_certified_predictor_curvature_second_directional(
    counts: NDArray,
    eta: NDArray,
    direction_a: NDArray,
    direction_b: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The packed second curvature direction and an entrywise bound on its error.

    ``-sum_kl l_ijkl a_k b_l`` under the two log links; directions ``(n, 2)``
    or stacked ``(m, n, 2)``.  ``l_bbbb`` enters only the ``bb`` channel, times
    ``a_b b_b``, so the certificate is ``|a_b b_b| e`` there, rounded upward
    once, and zero on ``aa`` and ``ab``.
    """
    from superglm.distributional.kernels._curvature_direction import curvature_second_direction

    predictors = np.asarray(eta, dtype=_FLOAT)
    first = np.asarray(direction_a, dtype=_FLOAT)
    second = np.asarray(direction_b, dtype=_FLOAT)
    expected = (len(np.asarray(counts)), 2)
    if (
        predictors.shape != expected
        or first.shape != second.shape
        or first.shape[-1] != 2
        or first.ndim not in (2, 3)
        or first.shape[-2] not in ((expected[0],) if first.ndim == 2 else (expected[0], 1))
        or not np.all(np.isfinite(predictors))
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError(
            "NB2 predictor state must be a finite n-by-2 array and the directions finite "
            "n-by-2 or m-by-n-by-2 arrays of one shape"
        )
    with np.errstate(over="ignore", under="ignore"):
        mean = np.exp(predictors[:, 0])
        theta = np.exp(predictors[:, 1])
    fourth, error = negative_binomial_log_fourth_derivatives(
        counts, mean, theta, weights, semantics
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = curvature_second_direction(fourth, first, second)
        certificate = np.zeros_like(result)
        certificate[..., 2] = np.nextafter(np.abs(first[..., 1] * second[..., 1]) * error, np.inf)
    _finite_output(
        result, "curvature second direction", NegativeBinomialDerivativeRepresentationError
    )
    _finite_output(
        certificate,
        "curvature-second-direction bound",
        NegativeBinomialDerivativeRepresentationError,
    )
    return (
        readonly(result, dtype=np.dtype(np.float64), shape=result.shape),
        readonly(certificate, dtype=np.dtype(np.float64), shape=certificate.shape),
    )


# Per-row term cap of the expected-information series: the series needs about
# ``mu + 37/log1p(theta/mu)`` terms, and the cap bounds one row's cost.
_MAX_INFORMATION_TERMS_PER_ROW = 1 << 26


def negative_binomial_expected_information(
    mean: NDArray,
    theta: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    """Return packed natural-scale NB2 Fisher information, one row per observation.

    Mean and size are information-orthogonal, so the packed ``(mean, size)``
    entry is zero (Lawless, *Canadian J. Statist.* 15(3), 1987, eqs. 2.7-2.8).
    Per unit mass ``I_mean = theta / (mu (mu + theta))`` and ``I_size`` is the
    series of ``_negative_binomial_numba._size_information_row``. Weights
    enter as the likelihood does: frequency weights multiply a row, prior
    weights rescale its mean and size. A row whose series would exceed the
    term budget, or whose information is not representable, raises
    ``NegativeBinomialNumericalDomainError``.
    """
    mean_values, theta_values, weight_values = (
        np.asarray(values, dtype=_FLOAT) for values in (mean, theta, weights)
    )
    scale, multiplier, effective_mean, effective_theta, total, poisson = _domain(
        mean_values, theta_values, weight_values, semantics
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        predicted = effective_mean + 37.0 / np.log1p(effective_theta / effective_mean) + 64.0
    # Only the per-row cap applies: the series is row-local and allocates
    # nothing per term, so the total work is what the rows need, and an
    # aggregate cap on the prediction (64 terms or more a row) would refuse a
    # call by its row count alone.
    if not np.all(np.isfinite(predicted)) or float(np.max(predicted)) > (
        _MAX_INFORMATION_TERMS_PER_ROW
    ):
        raise NegativeBinomialNumericalDomainError(
            "NB2 expected information needs more series terms than its budget allows"
        )
    size_information, terms = _compiled.expected_size_information(
        np.ascontiguousarray(effective_mean),
        np.ascontiguousarray(effective_theta),
        _MAX_INFORMATION_TERMS_PER_ROW,
    )
    if np.any(terms < 0):
        raise NegativeBinomialNumericalDomainError(
            "NB2 expected information series did not reach its truncation bound"
        )
    information = np.zeros((len(mean_values), 3), dtype=_FLOAT)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        information[:, 0] = multiplier * (scale * (effective_theta / total)) / mean_values
        information[:, 2] = multiplier * ((size_information / theta_values) / theta_values)
    # A Poisson-limit row carries no size information (see _POISSON_LIMIT_RATIO).
    information[poisson, 2] = 0.0
    if not np.all(np.isfinite(information)):
        raise NegativeBinomialNumericalDomainError("NB2 expected information is not representable")
    return readonly(information, dtype=np.dtype(np.float64), shape=(len(mean_values), 3))


__all__ = [
    "NegativeBinomialDerivativeRepresentationError",
    "NegativeBinomialInitializationError",
    "NegativeBinomialNumericalDomainError",
    "NegativeBinomialPoissonBoundaryError",
    "NegativeBinomialRowEvaluation",
    "evaluate_negative_binomial_rows",
    "has_resolved_poisson_boundary",
    "initialize_negative_binomial",
    "negative_binomial_certified_predictor_curvature_directional",
    "negative_binomial_certified_predictor_curvature_second_directional",
    "negative_binomial_expected_information",
    "negative_binomial_log_fourth_derivatives",
    "negative_binomial_log_third_derivatives",
    "recurrence_cells",
]


def _warmup_negative_binomial() -> None:
    """Compile the NB2 kernels a SuperLSS fit calls on its way (no rows).

    The theta-series core and the expected size information, which the
    terminal curvature policy falls back to and ``coefficient_curvature=
    "fisher"`` steps with. The LAML endgame's third- and fourth-derivative
    kernels compile on first use, as the Tweedie family's do.
    """
    rows = np.zeros(0, dtype=np.int64)
    values = np.zeros(0, dtype=_FLOAT)
    _compiled.theta_series_serial(
        rows, rows, values, values, values, values, values, 2, _SERIES_ORDER,
        _SERIES_EXPONENTS, _SERIES_BERNOULLI, np.zeros((0, 2), dtype=np.bool_),
        np.zeros((0, 2), dtype=_FLOAT), np.zeros((0, 2), dtype=_FLOAT),
    )  # fmt: skip
    _compiled._size_information_rows(
        values, values, _MAX_INFORMATION_TERMS_PER_ROW, 0, 0, values, rows
    )
