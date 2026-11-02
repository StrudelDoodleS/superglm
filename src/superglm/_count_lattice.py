"""Exact integrality test for supplied count arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _not_a_whole_number(values: NDArray) -> NDArray:
    """Which entries of a SUPPLIED array are not exactly whole numbers.

    For a declared replication count, and for a counting response under the
    frequency contract.  Both are values the caller hands over and declares to
    be counts; nothing here forms them, so there is no round-off to forgive and
    an exact test is the honest one.  ``counts.astype(float)`` is exactly
    integral, as is anything read from a count column.

    Exact rather than a ulp or two on purpose.  At ``2**49`` a ulp is already
    ``0.125`` of a count, so any non-zero allowance admits representable
    fractional counts at large magnitudes.  A count that is not exactly an
    integer is not a count, and saying so is more useful than guessing which
    near-integers were meant.
    """

    return values != np.rint(values)


__all__ = ["_not_a_whole_number"]
