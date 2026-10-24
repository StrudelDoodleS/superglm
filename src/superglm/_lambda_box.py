"""The box the REML engines search smoothing parameters in."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

#: The smallest smoothing parameter an engine proposes.
LAMBDA_FLOOR = 1.0e-6
#: The largest, for a penalty on coefficients in the units of the data
#: curvature it is weighed against.  A SCOP block's ceiling is
#: :func:`latent_lambda_ceiling`.
LAMBDA_CEILING = 1.0e10


def latent_lambda_ceiling(latent: NDArray) -> float:
    """The ceiling for a penalty on the log-increments ``latent`` of a SCOP block.

    The penalty acts on the logs of the block's increments, which carry no
    units, while the block's data curvature in those coordinates is
    ``J' A J`` with ``J = diag(exp(latent))``: it grows as the squared
    increments, so the smoothing parameter that balances the two grows with
    them (as the square of the response's units under an identity link).
    The box was set at unit increments; the ceiling is
    :data:`LAMBDA_CEILING` times the block's mean squared increment where
    that exceeds one, and :data:`LAMBDA_CEILING` otherwise.
    """
    doubled = 2.0 * np.asarray(latent, dtype=np.float64)
    if doubled.size == 0 or not np.all(np.isfinite(doubled)):
        return LAMBDA_CEILING
    shift = float(np.max(doubled))
    log_mean_square = shift + math.log(float(np.mean(np.exp(doubled - shift))))
    if log_mean_square <= 0.0:
        return LAMBDA_CEILING
    log_ceiling = math.log(LAMBDA_CEILING) + log_mean_square
    if log_ceiling >= math.log(np.finfo(np.float64).max):
        return float(np.finfo(np.float64).max)
    return math.exp(log_ceiling)
