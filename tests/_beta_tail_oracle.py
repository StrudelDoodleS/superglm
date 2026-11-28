"""An mpmath oracle for the incomplete beta function's log tails, for whole ``a`` and ``b``.

``I_x(a, b) = P(Binomial(a + b - 1, x) >= a)``, so the lower tail is a finite binomial sum,
read here at 60 digits. Below the mean ``a / (a + b)`` its terms fall from ``j = a`` with a
ratio ``r_j = (N - j) x / ((j + 1) (1 - x))`` that falls too, so the sum stops once the rest,
at most ``term r / (1 - r)``, is below ``1e-45`` of the total. ``x`` is an mpmath number, so a
caller reads the upper tail ``1 - I_u(a, b) = I_{1-u}(b, a)`` at ``1 - u`` exactly.
"""

from __future__ import annotations

import mpmath as mp


def log_beta_tail(a: int, b: int, x) -> float:
    """``log I_x(a, b)`` for whole ``a, b >= 1`` and ``x`` below ``(a + 1) / (a + b)``."""
    with mp.workdps(60):
        x = mp.mpf(x)
        y = 1 - x
        n = a + b - 1
        assert 0 <= x < mp.mpf(a + 1) / (a + b), "the oracle's sum falls only below the mean"
        if x == 0:
            return float("-inf")
        term = mp.binomial(n, a) * x**a * y ** (n - a)
        total = term
        for j in range(a, n):
            ratio = (n - j) * x / ((j + 1) * y)
            term *= ratio
            total += term
            if ratio < 1 and term * ratio / (1 - ratio) < total * mp.mpf(10) ** -45:
                break
        return float(mp.log(total))
