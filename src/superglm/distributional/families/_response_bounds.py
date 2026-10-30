"""Truncated and censored rows on top of a family's complete-row channels.

A family evaluates every row as complete, then :func:`bounded_channels`
replaces each censored row by ``m log P(observed window)`` and subtracts
``m log P(truncation window)`` from each truncated row, where ``m`` is the
row's replication mass.  The family supplies only ``window(lower, upper,
rows)``: the unit-mass log window probability with its natural-parameter
score and packed Hessian at those rows' parameters.

A censored row has no density, so its parameter-independent carrier is zero
(:func:`censored_carrier`).  Third and fourth derivatives are not formed here:
a family refuses its analytic hooks for a bounded plan, and the LAML ladder
differences these analytic Hessians instead.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    ObservationContract,
    bounded_observation,
)
from superglm.distributional.kernels._response_tails import TailChannels
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import UnsupportedLikelihoodContractError

WindowFunction = Callable[[NDArray, NDArray, NDArray[np.intp]], TailChannels]


def bind_observation(
    observation: ObservationContract,
    n_rows: int,
    *,
    support_lower: float,
    family_name: str,
) -> ObservationContract:
    """The contract a bounds-aware family stores: complete, or bounds on its support."""
    if observation == COMPLETE_OBSERVATION:
        return observation
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
    return bounded_observation(observation.bounds.on_support(support_lower))


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


__all__ = [
    "TailChannels",
    "WindowFunction",
    "bind_observation",
    "bounded_channels",
    "censored_carrier",
    "refuse_bounded",
]
