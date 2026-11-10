"""Independent NB2 CRPS: the lattice sum ``sum_k (F(k) - 1{y <= k})^2`` in mpmath.

No production code is imported. The pmf comes from the ratio
``f(k + 1) / f(k) = r (k + theta) / (k + 1)``, ``r = mu / (mu + theta)``, from
``f(0) = (1 - r)^theta``. Survival probabilities are summed down from the
cut ``K``, never formed as ``1 - F``, so a rare claim's ``P(Y > 0)`` keeps its
digits. Past ``K`` every pmf ratio is at most ``q_K = max(r, r (K + theta) / (K + 1))
< 1``, so ``S(K) <= f(K) q_K / (1 - q_K)``. Leaving ``S(K)`` out of every
``S(k)`` and the cells past ``K`` out of the sum moves the score by at most
``S(K) (2 mu + (K + 1) S(K)) + S(K)^2 / (1 - q_K^2)``; the sum stops once that is
below ``10^-35`` of ``min(f(0), f(1))^2``, which the score exceeds (below ``y = 1``
it holds ``S(0)^2 >= f(1)^2``, above it ``F(0)^2 = f(0)^2`` over a whole cell).
"""

from __future__ import annotations

import mpmath as mp

_DIGITS = 60
_RELATIVE_TAIL = mp.mpf(10) ** -35


def nb2_crps(mean: float, size: float, response: float) -> float:
    """``CRPS(NB2(mean, size), response)`` for a real response, to binary64."""
    with mp.workdps(_DIGITS):
        mu, theta, y = mp.mpf(mean), mp.mpf(size), mp.mpf(response)
        r = mu / (mu + theta)
        pmf = [mp.exp(theta * mp.log(theta / (mu + theta)))]
        pmf.append(pmf[0] * r * theta)
        scale = min(pmf) ** 2
        floor_y = int(mp.floor(y))
        while True:
            k = len(pmf) - 1
            ratio = max(r, r * (k + theta) / (k + 1))
            if k >= max(floor_y, int(mu)) + 1 and ratio < 1:
                tail = pmf[-1] * ratio / (1 - ratio)  # bounds S(k)
                moved = tail * (2 * mu + (k + 1) * tail) + tail**2 / (1 - ratio**2)
                if moved < _RELATIVE_TAIL * scale:
                    break
            pmf.append(pmf[-1] * r * (k + theta) / (k + 1))
        cut = len(pmf) - 1
        survival = [mp.mpf(0)] * (cut + 1)  # S(k) less the uncounted tail past the cut
        for k in range(cut - 1, -1, -1):
            survival[k] = survival[k + 1] + pmf[k + 1]
        total = mp.mpf(0) if y >= 0 else -y
        cdf = mp.mpf(0)
        for k in range(cut):
            cdf += pmf[k]
            if k + 1 <= y:  # the whole cell [k, k + 1) lies below y
                total += cdf**2
            elif k >= y:  # the whole cell lies at or above y
                total += survival[k] ** 2
            else:  # y splits the cell
                total += (y - k) * cdf**2 + (k + 1 - y) * survival[k] ** 2
        return float(total)


__all__ = ["nb2_crps"]
