"""Row-local normalized Tweedie point likelihood for distributional models.

This module implements a deterministic float64 point evaluator.  Its
``log_cutoff`` window is not an outward tail certificate and its refusals do
not imply that the mathematical Tweedie law is undefined.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.kernels._tweedie_numba as _compiled
from superglm._parallel import pooled_row_blocks
from superglm.distributional.kernels._common import (
    WeightSemantics,
    readonly,
    validated_derivative_order,
)

_MAX_SAFE_MODE = _compiled._MAX_SAFE_MODE
_MAX_EXACT_FREQUENCY = 2**53
_MIN_POWER = 1.05
_MAX_POWER = 1.95
_DEFAULT_MAX_TERMS = 100_000
_DEFAULT_LOG_CUTOFF = 37.0
# Below this many rows a batch runs as one serial block; above it, blocks of
# _PARALLEL_BLOCK rows run on the worker pool.
_PARALLEL_MIN_ROWS = 5_000
_PARALLEL_BLOCK = 4096

_RowsKernel = Callable[..., tuple[int, int]]
_FillKernel = Callable[..., None]


class TweedieNumericalRefusal(RuntimeError):  # noqa: N818 - binding contract name
    """The row is mathematically supported but unsafe for this point route."""


@dataclass(frozen=True)
class TweediePointEvaluation:
    """Complete row likelihood and requested natural-coordinate derivatives."""

    log_likelihood: NDArray[np.float64]
    score: NDArray[np.float64] | None
    hessian_packed: NDArray[np.float64] | None
    terms: NDArray[np.integer]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        log_likelihood = np.asarray(self.log_likelihood)
        if log_likelihood.ndim != 1 or not np.all(np.isfinite(log_likelihood)):
            raise ValueError("log_likelihood must be a finite one-dimensional array")
        n_rows = len(log_likelihood)
        owned_log_likelihood = readonly(log_likelihood, dtype=np.dtype(np.float64), shape=(n_rows,))
        owned_terms = readonly(self.terms, dtype=np.dtype(np.int64), shape=(n_rows,))
        if np.any(owned_terms < 0):
            raise ValueError("terms must be non-negative")
        owned_valid = readonly(self.valid, dtype=np.dtype(np.bool_), shape=(n_rows,))

        owned_score = None
        if self.score is not None:
            score = np.asarray(self.score)
            if not np.all(np.isfinite(score)):
                raise ValueError("score must contain only finite values")
            owned_score = readonly(score, dtype=np.dtype(np.float64), shape=(n_rows, 3))
        owned_hessian = None
        if self.hessian_packed is not None:
            hessian = np.asarray(self.hessian_packed)
            if not np.all(np.isfinite(hessian)):
                raise ValueError("hessian_packed must contain only finite values")
            owned_hessian = readonly(hessian, dtype=np.dtype(np.float64), shape=(n_rows, 6))

        object.__setattr__(self, "log_likelihood", owned_log_likelihood)
        object.__setattr__(self, "score", owned_score)
        object.__setattr__(self, "hessian_packed", owned_hessian)
        object.__setattr__(self, "terms", owned_terms)
        object.__setattr__(self, "valid", owned_valid)


def initialize_tweedie(
    response: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    *,
    power_lower: float,
    power_upper: float,
) -> NDArray[np.float64]:
    """Return starting mean and scale values for a Tweedie fit."""

    response_values = _strict_float64_vector(response, name="response")
    weight_values = _strict_float64_vector(weights, name="weights")
    if not response_values.size:
        raise ValueError("Tweedie initialization arrays must be non-empty")
    if response_values.shape != weight_values.shape:
        raise ValueError("Tweedie initialization arrays must have the same shape")
    if not np.all(np.isfinite(response_values)) or np.any(response_values < 0.0):
        raise ValueError("response must be finite and non-negative")
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    if semantics not in ("prior", "frequency"):
        raise ValueError("semantics must be 'prior' or 'frequency'")
    if semantics == "frequency" and (
        np.any(weight_values != np.floor(weight_values))
        or np.any(weight_values > _MAX_EXACT_FREQUENCY)
    ):
        raise ValueError("frequency weights must be exact positive integer replication counts")
    if isinstance(power_lower, (bool, np.bool_)) or isinstance(power_upper, (bool, np.bool_)):
        raise ValueError("power walls must be finite and strictly ordered inside (1, 2)")
    try:
        lower = float(power_lower)
        upper = float(power_upper)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("power walls must be finite and strictly ordered inside (1, 2)") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or not 1.0 < lower < upper < 2.0:
        raise ValueError("power walls must be finite and strictly ordered inside (1, 2)")
    if np.all(response_values == 0.0):
        raise ValueError("all-zero Tweedie samples have no finite interior initializer")
    weight_sum = float(np.sum(weight_values, dtype=np.float64))
    mean = float(np.dot(weight_values, response_values) / weight_sum)
    power = 0.5 * (lower + upper)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        scaled_squared_residual = (response_values - mean) ** 2 / mean**power
        numerator = float(np.dot(weight_values, scaled_squared_residual))
    denominator = len(response_values) if semantics == "prior" else weight_sum
    dispersion = numerator / denominator
    if (
        not math.isfinite(mean)
        or mean <= 0.0
        or not math.isfinite(dispersion)
        or dispersion <= 0.0
        or not lower < power < upper
    ):
        raise ValueError("Tweedie initialization must be finite and interior")
    theta = np.column_stack(
        (
            np.full(len(response_values), mean),
            np.full(len(response_values), dispersion),
            np.full(len(response_values), power),
        )
    )
    return readonly(theta, dtype=np.dtype(np.float64), shape=(len(response_values), 3))


def _strict_float64_vector(values: object, *, name: str) -> NDArray[np.float64]:
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be a numpy float64 array")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return values


def _exact_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validated_tweedie_inputs(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    semantics: WeightSemantics,
    *,
    derivative_order: int,
    max_terms: int = _DEFAULT_MAX_TERMS,
    log_cutoff: float = _DEFAULT_LOG_CUTOFF,
):
    """Validate the shared Python and compiled point-kernel boundary."""
    arrays = tuple(
        _strict_float64_vector(values, name=name)
        for values, name in (
            (y, "y"),
            (mean, "mean"),
            (dispersion, "dispersion"),
            (power, "power"),
            (weight, "weight"),
        )
    )
    y_values, mean_values, dispersion_values, power_values, weight_values = arrays
    if not y_values.size:
        raise ValueError("Tweedie row arrays must be non-empty")
    if any(values.shape != y_values.shape for values in arrays[1:]):
        raise ValueError("Tweedie row arrays must have the same shape")
    if not np.all(np.isfinite(y_values)) or np.any(y_values < 0.0):
        raise ValueError("y must be finite with y >= 0")
    if not np.all(np.isfinite(mean_values)) or np.any(mean_values <= 0.0):
        raise ValueError("mean must be finite and strictly positive")
    if not np.all(np.isfinite(dispersion_values)) or np.any(dispersion_values <= 0.0):
        raise ValueError("dispersion must be finite and strictly positive")
    if (
        not np.all(np.isfinite(power_values))
        or np.any(power_values <= 1.0)
        or np.any(power_values >= 2.0)
    ):
        raise ValueError("power must be finite and strictly between 1 and 2")
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0.0):
        raise ValueError("weight must be finite and strictly positive")
    if semantics not in ("prior", "frequency"):
        raise ValueError("semantics must be 'prior' or 'frequency'")
    if semantics == "frequency" and (
        np.any(weight_values != np.floor(weight_values))
        or np.any(weight_values > _MAX_EXACT_FREQUENCY)
    ):
        raise ValueError("frequency weights must be exact positive integer replication counts")

    order = validated_derivative_order(derivative_order)
    row_max_terms = _exact_nonnegative_integer(max_terms, name="max_terms")
    if isinstance(log_cutoff, (bool, np.bool_)):
        raise ValueError("log_cutoff must be finite and strictly positive")
    cutoff = float(log_cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("log_cutoff must be finite and strictly positive")
    if np.any(power_values < _MIN_POWER) or np.any(power_values > _MAX_POWER):
        raise TweedieNumericalRefusal(
            "row lies outside the inclusive certified power range [1.05, 1.95]"
        )
    return arrays, order, row_max_terms, cutoff


_COMPILED_ROW_REFUSALS = {
    _compiled.KERNEL_MODE_RATIO: "series mode ratio is not representable",
    _compiled.KERNEL_MODE_RANGE: "series mode lies above the exact float64 integer range 2**52",
    _compiled.KERNEL_MODE_BRACKET: "series mode could not be ratio-bracketed",
    _compiled.KERNEL_WINDOW_RANGE: "series window exceeds exact float64 integer work",
    _compiled.KERNEL_UPPER_RATIO: "upper series ratio lost its mode bracket",
    _compiled.KERNEL_LOWER_RATIO: "lower series ratio lost its mode bracket",
    _compiled.KERNEL_PEAK: "series peak is not representable",
    _compiled.KERNEL_MASS: "series mass is not representable",
    _compiled.KERNEL_SCORE_MOMENTS: "series score moments are not representable",
    _compiled.KERNEL_HESSIAN_MOMENTS: "series Hessian moments are not representable",
    _compiled.KERNEL_MEAN_SCORE_SCALE: "mean score scale is not representable",
    _compiled.KERNEL_MEAN_SCORE: "mean score channel is not representable",
    _compiled.KERNEL_MEAN_HESSIAN_SCALE: "mean Hessian scale is not representable",
    _compiled.KERNEL_MEAN_HESSIAN: "mean Hessian channel is not representable",
    _compiled.KERNEL_ZERO_RATE: "zero-atom rate is not representable",
    _compiled.KERNEL_CANONICAL_SCALE: "positive-row canonical scale is not representable",
    _compiled.KERNEL_SERIES_BASE: "positive-row series base is not representable",
    _compiled.KERNEL_ROW_VALUE: "positive row value is not representable",
    _compiled.KERNEL_ROW_SCORE: "positive row score is not representable",
    _compiled.KERNEL_ROW_DERIVATIVES: "positive row derivatives are not representable",
    _compiled.KERNEL_REQUIRED_WORK: "required float64 work is not representable",
}

_evaluate_tweedie_batch_rows = cast(_RowsKernel, _compiled._evaluate_tweedie_batch_rows)


def _raise_compiled_refusal(*, status: int, failing_row: int, max_terms: int) -> None:
    complete_messages: dict[int | np.int64, str] = {
        _compiled.KERNEL_COMPLETE_VALUE: "complete row value is not representable",
        _compiled.KERNEL_COMPLETE_SCORE: "complete row score is not representable",
        _compiled.KERNEL_COMPLETE_HESSIAN: "complete row Hessian is not representable",
    }
    if status in complete_messages:
        raise TweedieNumericalRefusal(complete_messages[status])
    if status == _compiled.KERNEL_MAX_TERMS:
        message = f"positive series window reached per-row max_terms={max_terms}"
    else:
        message = _COMPILED_ROW_REFUSALS.get(status)
    if message is None or failing_row < 0:
        raise RuntimeError(f"unknown compiled Tweedie kernel status {status}")
    raise TweedieNumericalRefusal(f"row {failing_row}: {message}")


# The LAML endgame's third- and fourth-derivative kernels
# (``_tweedie_hessian_direction_rows``, ``_tweedie_fourth_rows`` and the
# helpers they call) compile, or load from numba's cache, on a Tweedie fit's
# first endgame call, not in ``superglm.warmup()``: only a fit whose
# smoothing parameter reaches its endpoint calls them (the first-use list in
# tests/test_import_compat.py records the measurements).


def _pooled_rows(kernel: Callable[[int, int], tuple[int, int]], n_rows: int) -> tuple[int, int]:
    """Run ``kernel(start, stop)`` over the rows: one block below ``_PARALLEL_MIN_ROWS``,
    else blocks of ``_PARALLEL_BLOCK`` on the worker pool.  The first failure in
    row order, ``(status, row)``, or ``(KERNEL_OK, -1)``."""
    if n_rows < _PARALLEL_MIN_ROWS:
        return kernel(0, n_rows) if n_rows else (int(_compiled.KERNEL_OK), -1)
    for status, row in pooled_row_blocks(kernel, n_rows, _PARALLEL_BLOCK):
        if status != _compiled.KERNEL_OK:
            return status, row
    return int(_compiled.KERNEL_OK), -1


def _first_failure(statuses: NDArray[np.int64]) -> tuple[int, int]:
    """``(status, row)`` of the first row whose status is not ``KERNEL_OK``, else ``(OK, -1)``."""
    failing = np.flatnonzero(statuses != _compiled.KERNEL_OK)
    if failing.size:
        return int(statuses[failing[0]]), int(failing[0])
    return int(_compiled.KERNEL_OK), -1


def _batch_outputs(n_rows: int, order: int) -> tuple[NDArray, ...]:
    """``(values, scores, hessians, terms, valid)`` for ``n_rows`` rows at derivative ``order``."""
    return (
        np.empty(n_rows, dtype=np.float64),
        np.empty((n_rows if order >= 1 else 0, 3), dtype=np.float64),
        np.empty((n_rows if order == 2 else 0, 6), dtype=np.float64),
        np.empty(n_rows, dtype=np.int64),
        np.ones(n_rows, dtype=np.bool_),
    )


def _warmup_tweedie() -> None:
    """Compile the batch's row kernel; every helper it calls compiles with it."""
    arrays = tuple(np.array([value], dtype=np.float64) for value in (1.0, 1.2, 0.7, 1.5, 1.0))
    values, scores, hessians, terms, _ = _batch_outputs(1, 2)
    status, failing_row = _evaluate_tweedie_batch_rows(
        *arrays,
        0,
        2,
        _DEFAULT_MAX_TERMS,
        _DEFAULT_LOG_CUTOFF,
        0,
        1,
        values,
        scores,
        hessians,
        terms,
    )
    if status == _compiled.KERNEL_OK:
        status = _compiled._complete_batch_status(values, scores, hessians, 2)
    if status != _compiled.KERNEL_OK:
        _raise_compiled_refusal(
            status=int(status), failing_row=int(failing_row), max_terms=_DEFAULT_MAX_TERMS
        )


def evaluate_tweedie_rows(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    semantics: WeightSemantics,
    *,
    derivative_order: int,
    max_terms: int = _DEFAULT_MAX_TERMS,
    log_cutoff: float = _DEFAULT_LOG_CUTOFF,
) -> TweediePointEvaluation:
    """Evaluate independent normalized Tweedie rows in natural coordinates.

    The accepted point range is inclusive ``1.05 <= p <= 1.95``.  Supported
    mixed-law rows nearer ``1`` or ``2`` raise :class:`TweedieNumericalRefusal`
    because no defended endpoint route exists.
    """
    arrays, order, row_max_terms, cutoff = _validated_tweedie_inputs(
        y,
        mean,
        dispersion,
        power,
        weight,
        semantics,
        derivative_order=derivative_order,
        max_terms=max_terms,
        log_cutoff=log_cutoff,
    )
    normalized = tuple(
        np.require(values, dtype=np.float64, requirements=("C", "A", "W")) for values in arrays
    )
    # No representable series window can contain more than 2**52 terms.  Keep
    # the public Python cap for refusal messages, but never dispatch an
    # oversized Python integer through Numba's native integer boundary.
    native_max_terms = min(row_max_terms, _MAX_SAFE_MODE + 1)
    n_rows = len(normalized[0])
    values, scores, hessians, terms, valid = _batch_outputs(n_rows, order)
    weight_mode = 0 if semantics == "prior" else 1
    status, failing_row = _pooled_rows(
        lambda start, stop: _evaluate_tweedie_batch_rows(
            *normalized, weight_mode, order, native_max_terms, cutoff,
            start, stop, values, scores, hessians, terms,
        ),
        n_rows,
    )  # fmt: skip
    if status == _compiled.KERNEL_OK:
        status = _compiled._complete_batch_status(values, scores, hessians, order)
    if status != _compiled.KERNEL_OK:
        _raise_compiled_refusal(
            status=int(status),
            failing_row=int(failing_row),
            max_terms=row_max_terms,
        )
    return TweediePointEvaluation(
        log_likelihood=values,
        score=scores if order >= 1 else None,
        hessian_packed=hessians if order == 2 else None,
        terms=terms,
        valid=valid,
    )


def tweedie_hessian_direction(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    semantics: WeightSemantics,
    direction: NDArray[np.float64],
    *,
    max_terms: int = _DEFAULT_MAX_TERMS,
    log_cutoff: float = _DEFAULT_LOG_CUTOFF,
) -> NDArray[np.float64]:
    """Packed derivative of the natural ``(mu, phi, p)`` Hessian along ``direction``.

    The positive-row series contributes the joint cumulants of its term
    derivatives over the Hessian's own window (two passes: means, then
    central moments about them); the zero row and the canonical part are
    closed forms.  Rows the Hessian would refuse are refused here too.

    The cumulants of the window's normalized terms are exactly the derivatives
    of the windowed sums, so this is the derivative of the published Hessian,
    window included, which is the curvature the LAML objective is defined on;
    it carries no certificate for the omitted tail, and needs none against
    that objective.  The window's distance from the full Tweedie law is the
    published likelihood's own, uncertified (see the module note), and
    v0.40.0's Richardson stencil differenced the same windowed Hessian, so
    its certificate did not cover that distance either.  The Hessian moves
    discontinuously only where the window gains or loses a term below
    ``exp(-log_cutoff)`` of the mode term: at the default cutoff, within the
    Hessian's own rounding.
    """
    arrays, _, row_max_terms, cutoff = _validated_tweedie_inputs(
        y,
        mean,
        dispersion,
        power,
        weight,
        semantics,
        derivative_order=2,
        max_terms=max_terms,
        log_cutoff=log_cutoff,
    )
    natural = np.require(np.asarray(direction, dtype=np.float64), requirements=("C", "A"))
    if natural.shape != (len(arrays[0]), 3) or not np.all(np.isfinite(natural)):
        raise ValueError("Tweedie natural direction must be a finite n-by-3 array")
    normalized = tuple(
        np.require(values, dtype=np.float64, requirements=("C", "A", "W")) for values in arrays
    )
    n_rows = len(normalized[0])
    result = np.empty((n_rows, 6), dtype=np.float64)
    statuses = np.empty(n_rows, dtype=np.int64)
    weight_mode = 0 if semantics == "prior" else 1
    native_max_terms = min(row_max_terms, _MAX_SAFE_MODE + 1)
    pooled_row_blocks(
        lambda start, stop: cast(_FillKernel, _compiled._tweedie_hessian_direction_rows)(
            *normalized, weight_mode, natural, native_max_terms, cutoff,
            start, stop, result, statuses,
        ),
        n_rows,
        _PARALLEL_BLOCK,
    )  # fmt: skip
    status, failing_row = _first_failure(statuses)
    if status != _compiled.KERNEL_OK:
        _raise_compiled_refusal(
            status=int(status), failing_row=int(failing_row), max_terms=row_max_terms
        )
    return result


def tweedie_natural_fourth(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    semantics: WeightSemantics,
    *,
    max_terms: int = _DEFAULT_MAX_TERMS,
    log_cutoff: float = _DEFAULT_LOG_CUTOFF,
) -> NDArray[np.float64]:
    """Derivatives of orders one to four in ``(mu, rho, p)``, ``rho = log phi``, as ``(n, 34)``.

    Packed by order (3 + 6 + 10 + 15).  ``rho`` rather than ``phi`` because
    the series is linear in ``rho`` and the default dispersion link is
    ``rho`` itself; the caller composes with its links.  The positive row's
    series contributes the joint cumulants of its term derivatives through
    fourth order over the Hessian's own window (two passes, as
    :func:`tweedie_hessian_direction`), so this differentiates the published,
    windowed likelihood; the closed part, and the zero row, are generated
    closed forms.  Rows the Hessian would refuse are refused here too.
    """
    arrays, _, row_max_terms, cutoff = _validated_tweedie_inputs(
        y,
        mean,
        dispersion,
        power,
        weight,
        semantics,
        derivative_order=2,
        max_terms=max_terms,
        log_cutoff=log_cutoff,
    )
    normalized = tuple(
        np.require(values, dtype=np.float64, requirements=("C", "A", "W")) for values in arrays
    )
    n_rows = len(normalized[0])
    result = np.empty((n_rows, 34), dtype=np.float64)
    statuses = np.empty(n_rows, dtype=np.int64)
    weight_mode = 0 if semantics == "prior" else 1
    native_max_terms = min(row_max_terms, _MAX_SAFE_MODE + 1)
    pooled_row_blocks(
        lambda start, stop: cast(_FillKernel, _compiled._tweedie_fourth_rows)(
            *normalized, weight_mode, native_max_terms, cutoff, start, stop, result, statuses
        ),
        n_rows,
        _PARALLEL_BLOCK,
    )
    status, failing_row = _first_failure(statuses)
    if status != _compiled.KERNEL_OK:
        _raise_compiled_refusal(
            status=int(status), failing_row=int(failing_row), max_terms=row_max_terms
        )
    return result


__all__ = [
    "TweedieNumericalRefusal",
    "TweediePointEvaluation",
    "initialize_tweedie",
    "evaluate_tweedie_rows",
    "tweedie_hessian_direction",
    "tweedie_natural_fourth",
]
