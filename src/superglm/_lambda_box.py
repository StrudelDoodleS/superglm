"""The box the REML engines search smoothing parameters in."""

from __future__ import annotations

import math

import numpy as np

from superglm._roundoff import TINY

#: The smallest smoothing parameter an engine proposes, and the largest, for a
#: penalty on coefficients in the units of the data curvature it is weighed
#: against.  A SCOP block's box is :func:`scaled_lambda_box`.
LAMBDA_FLOOR = 1.0e-6
LAMBDA_CEILING = 1.0e10
LAMBDA_BOX = (LAMBDA_FLOOR, LAMBDA_CEILING)

_LARGEST = float(np.finfo(np.float64).max)


def scaled_lambda_box(scale: float) -> tuple[float, float]:
    """The ``(floor, ceiling)`` for a smoothing parameter that carries the units of ``scale``.

    A penalty on coefficients that carry no units, such as the logs of a SCOP
    block's increments, is weighed against a deviance that carries the
    dispersion's, so its smoothing parameter does too: the REML fixed point
    ``phi * rEDF / pSp`` (Wood and Fasiolo 2017) moves with ``phi``.  The box
    was set where that scale is one; it widens by ``scale``: the ceiling is
    :data:`LAMBDA_CEILING` times ``scale`` where ``scale`` exceeds one, the
    floor :data:`LAMBDA_FLOOR` times ``scale`` where it is below one, so either
    end follows the units and the box never narrows.  Each end stays within
    the positive normal binary64 range, and a ``scale`` that is not a positive
    finite number has :data:`LAMBDA_BOX`.
    """
    if not (math.isfinite(scale) and scale > 0.0):
        return LAMBDA_BOX
    if scale > 1.0:
        return LAMBDA_FLOOR, min(LAMBDA_CEILING * scale, _LARGEST)
    if scale < 1.0:
        return max(LAMBDA_FLOOR * scale, TINY), LAMBDA_CEILING
    return LAMBDA_BOX
