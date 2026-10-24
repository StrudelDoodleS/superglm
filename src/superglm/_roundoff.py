"""Binary64 rounding constants, Higham's gamma_n and the eps-power thresholds, stated once.

``UNIT_ROUNDOFF`` is u = 2^-53, the bound on the relative error of one
rounding to nearest (Higham 2002, sec. 2.1).  ``EPS`` is 2u = 2^-52, the
spacing of the binary64 numbers at one (``np.finfo(np.float64).eps``).
``gamma(n)`` is gamma_n = n u / (1 - n u), the bound on n roundings
(Lemma 3.1).  A bound written in eps, n eps / (1 - n eps), is
``gamma(2 * n)``: the same number, since doubling is exact.
"""

from __future__ import annotations

import numpy as np

UNIT_ROUNDOFF = 2.0**-53
EPS = 2.0**-52
#: The smallest normal binary64 number.
TINY = 2.0**-1022
#: The smallest positive binary64 number (subnormal).
SMALLEST_SUBNORMAL = 2.0**-1074
SQRT_EPS = EPS**0.5
#: A penalty eigenvalue below this fraction of the largest is null.
PENALTY_RANK_CUT = EPS ** (2.0 / 3.0)
#: The modified-Newton floor on a Hessian eigenvalue, relative to the largest.
NEWTON_EIGENVALUE_FLOOR = EPS**0.7


def gamma(count):
    """``gamma_count``; infinite once ``count u`` reaches one.  Elementwise on an array."""
    product = count * UNIT_ROUNDOFF
    if np.ndim(product):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(product < 1.0, product / (1.0 - product), np.inf)
    return product / (1.0 - product) if product < 1.0 else float("inf")


def gamma_up(count: float) -> float:
    """``gamma_count`` with its quotient rounded up (``count u`` is exact)."""
    product = float(count) * UNIT_ROUNDOFF
    if product >= 1.0:
        return float("inf")
    return float(np.nextafter(product / (1.0 - product), np.inf))
