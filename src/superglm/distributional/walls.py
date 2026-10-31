"""Rows of a bounded parameter near its walls, and how a coefficient step moves them."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import WallDisclosingFamily
from superglm.distributional.layout import PredictorState, StackedLayout

# Within this fraction of the distance between its walls, of either one, a
# bounded logit is in its exponential tail.
LINK_TAIL = 1.0e-2
# In that tail an exact Newton step moves a row the data press outward by up
# to one unit of its linear predictor, and by about one half or more even when
# the data's optimum lies just beyond the wall; at a finite optimum the step
# vanishes as the fit settles.
PRESSED_STEP = 0.25


class WallRows(NamedTuple):
    """The bounded parameter's fitted values, and which rows lie in the link's tail."""

    state: PredictorState
    lower: float
    upper: float
    values: NDArray[np.float64]
    upper_side: NDArray[np.bool_]
    near: NDArray[np.bool_]


def predictor_values(state: PredictorState, coefficients: NDArray) -> NDArray[np.float64]:
    """``X c`` for one predictor's coefficients, its intercept implicit and no offset."""
    intercept = state.intercept_index is not None
    values = np.full(state.design.n, coefficients[0] if intercept else 0.0, dtype=np.float64)
    if state.design.p:
        values += state.design.matvec(coefficients[int(intercept) :])
    return values


def wall_rows(
    family: WallDisclosingFamily,
    layout: StackedLayout,
    coefficients: NDArray,
    theta: NDArray | None,
) -> WallRows | None:
    """The bounded parameter's rows, or ``None`` when none lies within ``LINK_TAIL`` of a wall.

    ``theta`` holds the fitted parameters when the fit kept them; otherwise the
    bounded parameter is evaluated from its predictor and offset.
    """
    parameter_index, lower, upper = family.walls()
    state = next(state for state in layout.predictors if state.parameter_index == parameter_index)
    if theta is None:
        eta = predictor_values(state, coefficients[state.coefficient_slice]) + state.offset
        values = np.asarray(state.link.inverse(eta), dtype=np.float64)
    else:
        values = np.asarray(theta[:, parameter_index], dtype=np.float64)
    upper_side = upper - values <= values - lower
    distance = np.where(upper_side, upper - values, values - lower) / (upper - lower)
    near = distance <= LINK_TAIL
    if not np.any(near):
        return None
    return WallRows(state, lower, upper, values, upper_side, near)


def outward_step(rows: WallRows, step: NDArray) -> NDArray[np.float64]:
    """How far a full coefficient step moves each row's linear predictor towards its nearer wall."""
    move = predictor_values(rows.state, step[rows.state.coefficient_slice])
    return np.where(rows.upper_side, move, -move)
