"""Exact integrality test for supplied count arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _not_a_whole_number(values: NDArray) -> NDArray:
    """Which entries of a supplied count array are not exactly whole numbers.

    For declared replication counts and counting responses: the caller hands
    them over as counts, so there is no round-off to forgive. The test is
    exact because at ``2**49`` a ulp is already ``0.125`` of a count, so any
    allowance would admit fractional counts at large magnitudes.
    """

    return values != np.rint(values)


__all__ = ["_not_a_whole_number"]
