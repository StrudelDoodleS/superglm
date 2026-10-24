"""The box the REML engines search smoothing parameters in."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import TINY

#: The smallest smoothing parameter an engine proposes, and the largest, for a
#: penalty on coefficients in the units of the data curvature it is weighed
#: against.  A SCOP block's box is :func:`latent_lambda_box`.
LAMBDA_FLOOR = 1.0e-6
LAMBDA_CEILING = 1.0e10
LAMBDA_BOX = (LAMBDA_FLOOR, LAMBDA_CEILING)

_LOG_LARGEST = math.log(np.finfo(np.float64).max)
_LOG_TINY = math.log(TINY)


def latent_lambda_box(latent: NDArray) -> tuple[float, float]:
    """The ``(floor, ceiling)`` for a penalty on the log-increments ``latent`` of a SCOP block.

    The penalty acts on the logs of the block's increments, which carry no
    units, while the block's data curvature in those coordinates is
    ``J' A J`` with ``J = diag(exp(latent))``: it grows as the squared
    increments, so the smoothing parameter that balances the two grows with
    them (as the square of the response's units under an identity link).
    The box was set at unit increments; it widens by the block's mean squared
    increment ``ms``: the ceiling is :data:`LAMBDA_CEILING` times ``ms`` where
    ``ms`` exceeds one, the floor :data:`LAMBDA_FLOOR` times ``ms`` where
    ``ms`` is below one, so either end follows the units and the box never
    narrows.  Each end stays within the positive normal binary64 range.
    """
    doubled = 2.0 * np.asarray(latent, dtype=np.float64)
    if doubled.size == 0 or not np.all(np.isfinite(doubled)):
        return LAMBDA_BOX
    shift = float(np.max(doubled))
    log_mean_square = shift + math.log(float(np.mean(np.exp(doubled - shift))))
    if log_mean_square > 0.0:
        log_ceiling = math.log(LAMBDA_CEILING) + log_mean_square
        if log_ceiling >= _LOG_LARGEST:
            return LAMBDA_FLOOR, float(np.finfo(np.float64).max)
        return LAMBDA_FLOOR, math.exp(log_ceiling)
    if log_mean_square < 0.0:
        log_floor = math.log(LAMBDA_FLOOR) + log_mean_square
        if log_floor <= _LOG_TINY:
            return TINY, LAMBDA_CEILING
        return math.exp(log_floor), LAMBDA_CEILING
    return LAMBDA_BOX
