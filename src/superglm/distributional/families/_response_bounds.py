"""Truncated and censored rows on top of a family's complete-row channels.

A family evaluates every row as complete, then :func:`bounded_channels`
replaces each censored row by ``m log P(observed window)`` and subtracts
``m log P(truncation window)`` from each truncated row, where ``m`` is the
row's replication mass.  The family supplies only ``window(lower, upper,
rows)``: the unit-mass log window probability with its natural-parameter
score and packed Hessian at those rows' parameters.

A censored row has no density, so its parameter-independent carrier is zero
(:func:`censored_carrier`).  Each family's window supplies packed jets to
order four, and its complete-row third and fourth derivatives are adjusted
the same way with :func:`bounded_higher_derivatives` (or
:func:`bounded_share` with :func:`contract_third` for a directional hook).
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    ObservationContract,
    bounded_observation,
)
from superglm.distributional.kernels._curvature_direction import packed_tuples
from superglm.distributional.kernels._response_tails import Jet, TailChannels
from superglm.distributional.observation import LEFT_CENSORED, ResponseBounds
from superglm.distributional.weights import UnsupportedLikelihoodContractError

WindowFunction = Callable[[NDArray, NDArray, NDArray[np.intp]], TailChannels]
#: ``window_jets(lower, upper, rows, order)``: the unit-mass log window
#: probability's packed jet to ``order`` at those rows' parameters.
WindowJetFunction = Callable[[NDArray, NDArray, NDArray[np.intp], int], Jet]


def bind_observation(
    observation: ObservationContract,
    response: NDArray,
    weights: object,
    *,
    support_lower: float,
    family_name: str,
) -> ObservationContract:
    """The contract a bounds-aware family stores: complete, or bounds on its support.

    Bounds left with no effect once read on the support make the contract
    complete, so the plan is the complete one.  Refusals name input rows
    through the resolved weights' ``input_positions``.
    """
    if observation == COMPLETE_OBSERVATION:
        return observation
    n_rows = len(response)
    if (
        not isinstance(observation, ObservationContract)
        or observation.kind != "censored_truncated"
        or observation.bounds is None
    ):
        raise UnsupportedLikelihoodContractError(
            f"{family_name} supports complete and censored/truncated observations only"
        )
    if observation.bounds.n_rows != n_rows:
        raise UnsupportedLikelihoodContractError(
            f"{family_name} response rows do not match the response-bound rows"
        )
    bounds = observation.bounds.on_support(support_lower)
    impossible = (bounds.upper <= support_lower) | (
        (bounds.side == LEFT_CENSORED) & (response <= support_lower)
    )
    if np.any(impossible):
        positions = getattr(weights, "input_positions", None)
        first = int(np.flatnonzero(impossible)[0])
        if positions is not None and len(positions) == n_rows:
            first = int(positions[first])
        raise ValueError(
            f"{int(np.count_nonzero(impossible))} row(s) are censored or truncated at or below "
            f"the start of the {family_name} response support, so they have zero probability "
            f"(first at input row {first}); drop those bounds or those rows"
        )
    if len(bounds.truncated_rows()) == 0 and len(bounds.censored_rows()) == 0:
        return COMPLETE_OBSERVATION
    return bounded_observation(bounds)


def _window_side(values: object, n_rows: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim > 1 or array.size not in (1, n_rows):
        raise ValueError(f"{name} must be a number or one number per row ({n_rows} rows)")
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not be NaN; use -inf or inf for an open end")
    return np.broadcast_to(array.reshape(-1), (n_rows,))


def log_window_values(
    lower: NDArray,
    upper: NDArray,
    n_rows: int,
    window: WindowFunction,
    *,
    support_lower: float,
) -> NDArray[np.float64]:
    """``log P(lower < Y <= upper)`` per row; zero where the window holds the whole support.

    Each end is a number or one number per row, with ``lower <= upper``.  A
    lower end below the support's start reads as that start, and a window
    left empty, at or below the support, has log probability ``-inf``.
    """
    lo = _window_side(lower, n_rows, "lower")
    hi = _window_side(upper, n_rows, "upper")
    if np.any(lo > hi):
        raise ValueError("each window must have lower <= upper")
    lo = np.maximum(lo, support_lower)
    empty = lo >= hi
    values = np.where(empty, -np.inf, 0.0)
    rows = np.flatnonzero(~empty & ((lo > support_lower) | (hi < np.inf)))
    if len(rows):
        values[rows] = window(lo[rows], hi[rows], rows)[0]
    return values


def refuse_bounded(observation: ObservationContract, what: str) -> None:
    """Refuse a complete-row-only operation on a bounded plan.

    The exact error type matters: the LAML derivative ladder reads it as "no
    analytic derivative" and differences the bounded Hessian instead.
    """
    if not observation.is_complete:
        raise UnsupportedLikelihoodContractError(
            f"{what} is for complete rows; censored or truncated rows have none"
        )


def censored_carrier(carrier: NDArray, bounds: ResponseBounds | None) -> NDArray[np.float64]:
    """The complete-row carrier with censored rows, which have no density, set to zero."""
    values = np.asarray(carrier, dtype=np.float64)
    if bounds is None or len(bounds.censored_rows()) == 0:
        return values
    return np.where(bounds.side != 0, 0.0, values)


def bounded_channels(
    optimizing: NDArray,
    score: NDArray | None,
    hessian: NDArray | None,
    valid: NDArray,
    *,
    response: NDArray,
    bounds: ResponseBounds,
    multiplier: NDArray,
    window: WindowFunction,
    derivative_order: int,
) -> tuple[NDArray[np.float64], NDArray | None, NDArray | None, NDArray[np.bool_]]:
    """Complete-row channels adjusted for the plan's censored and truncated rows."""
    optimizing = np.array(optimizing, dtype=np.float64, copy=True)
    score = None if score is None else np.array(score, dtype=np.float64, copy=True)
    hessian = None if hessian is None else np.array(hessian, dtype=np.float64, copy=True)
    valid = np.array(valid, dtype=bool, copy=True)
    mass = np.asarray(multiplier, dtype=np.float64)

    censored = bounds.censored_rows()
    if len(censored):
        low, high = bounds.observed_window(response, censored)
        value, row_score, row_hessian = window(low, high, censored)
        m = mass[censored]
        optimizing[censored] = m * value
        if score is not None and row_score is not None:
            score[censored] = m[:, None] * row_score
        if hessian is not None and row_hessian is not None:
            hessian[censored] = m[:, None] * row_hessian
        valid[censored] = np.isfinite(value)

    truncated = bounds.truncated_rows()
    if len(truncated):
        value, row_score, row_hessian = window(
            bounds.lower[truncated], bounds.upper[truncated], truncated
        )
        m = mass[truncated]
        optimizing[truncated] -= m * value
        if score is not None and row_score is not None:
            score[truncated] -= m[:, None] * row_score
        if hessian is not None and row_hessian is not None:
            hessian[truncated] -= m[:, None] * row_hessian
        valid[truncated] &= np.isfinite(value)

    # Invalid rows still carry finite placeholders, as the evaluation requires.
    finite = np.isfinite(optimizing)
    for channels in (score, hessian):
        if channels is not None and derivative_order >= 1:
            finite &= np.all(np.isfinite(channels), axis=1)
    if not np.all(finite):
        valid &= finite
        optimizing[~finite] = 0.0
        for channels in (score, hessian):
            if channels is not None:
                channels[~finite] = 0.0
    return optimizing, score, hessian, valid


def bounded_higher_derivatives(
    third: NDArray | None,
    fourth: NDArray | None,
    valid: NDArray,
    *,
    response: NDArray,
    bounds: ResponseBounds,
    multiplier: NDArray,
    window_jets: WindowJetFunction,
) -> tuple[NDArray | None, NDArray | None, NDArray[np.bool_]]:
    """Complete-row packed third and fourth derivatives adjusted for bounded rows.

    The adjustment is :func:`bounded_channels`' one order up: a censored row's
    derivatives become ``m`` times the observed window's, and a truncated row
    loses ``m`` times its truncation window's.  The window jets must be in the
    coordinates of ``third`` and ``fourth``.  A row whose adjusted derivatives
    are not finite is zeroed and marked invalid.
    """
    arrays = [
        None if a is None else np.array(a, dtype=np.float64, copy=True) for a in (third, fourth)
    ]
    valid = np.array(valid, dtype=bool, copy=True)
    present = [index for index, values in enumerate(arrays) if values is not None]
    if not present:
        return None, None, valid
    order = 3 + max(present)
    mass = np.asarray(multiplier, dtype=np.float64)

    censored = bounds.censored_rows()
    if len(censored):
        low, high = bounds.observed_window(response, censored)
        jets = window_jets(low, high, censored, order)
        m = mass[censored][:, None]
        for index in present:
            arrays[index][censored] = m * jets[3 + index]

    truncated = bounds.truncated_rows()
    if len(truncated):
        jets = window_jets(bounds.lower[truncated], bounds.upper[truncated], truncated, order)
        m = mass[truncated][:, None]
        for index in present:
            arrays[index][truncated] -= m * jets[3 + index]

    finite = np.ones(len(valid), dtype=bool)
    for index in present:
        finite &= np.all(np.isfinite(arrays[index]), axis=1)
    if not np.all(finite):
        valid &= finite
        for index in present:
            arrays[index][~finite] = 0.0
    return arrays[0], arrays[1], valid


def bounded_share(
    order: int,
    k_parameters: int,
    *,
    response: NDArray,
    bounds: ResponseBounds,
    multiplier: NDArray,
    window_jets: WindowJetFunction,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Bounded rows' packed derivatives of ``order`` (3 or 4), each window formed once.

    Returns what :func:`bounded_higher_derivatives` adds to complete-row
    derivatives of zero, the rows whose complete-row derivatives stay (all but
    the censored), and the rows whose window is representable.
    """
    n_rows = len(response)
    zeros = np.zeros((n_rows, comb(k_parameters + order - 1, order)))
    third, fourth, valid = bounded_higher_derivatives(
        zeros if order == 3 else None,
        zeros if order == 4 else None,
        np.ones(n_rows, dtype=bool),
        response=response,
        bounds=bounds,
        multiplier=multiplier,
        window_jets=window_jets,
    )
    share = third if order == 3 else fourth
    assert share is not None
    return share, (bounds.side == 0) & valid, valid


def contract_third(third: NDArray, direction: NDArray) -> NDArray[np.float64]:
    """Packed ``sum_k T_ijk v_k`` from packed third derivatives and a direction ``(n, K)``."""
    k_parameters = direction.shape[1]
    column = {key: index for index, key in enumerate(packed_tuples(k_parameters, 3))}
    return np.column_stack(
        [
            sum(
                third[:, column[tuple(sorted((i, j, k)))]] * direction[:, k]
                for k in range(k_parameters)
            )
            for i, j in packed_tuples(k_parameters, 2)
        ]
    )


__all__ = [
    "LEFT_CENSORED",
    "Jet",
    "ResponseBounds",
    "TailChannels",
    "WindowFunction",
    "WindowJetFunction",
    "bind_observation",
    "bounded_channels",
    "bounded_higher_derivatives",
    "bounded_share",
    "censored_carrier",
    "contract_third",
    "log_window_values",
    "refuse_bounded",
]
