"""The box the REML engines search smoothing parameters in."""

from __future__ import annotations

#: The smallest smoothing parameter an engine proposes, and the largest, for a
#: penalty on coefficients in the units of the data curvature it is weighed
#: against.
LAMBDA_FLOOR = 1.0e-6
LAMBDA_CEILING = 1.0e10
LAMBDA_BOX = (LAMBDA_FLOOR, LAMBDA_CEILING)
