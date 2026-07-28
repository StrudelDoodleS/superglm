"""Penalized efficient-score statistic for one candidate pair.

Given the pair's cell-assembled score ``U`` and curvature ``V`` (Task 1), the
overlap cross-moments ``C``/``M`` against the span the mains already fit, and
the pair's tensor penalty ``S``, the statistic is

    T = U_eff' (V_eff + lambda0 * S)^{-1} U_eff

with the efficient-score adjustments ``U_eff = U - C' M^{-1} u_m`` and
``V_eff = V - C' M^{-1} C``, and ``lambda0`` chosen so the smooth is compared
at a fixed screening complexity: ``tr((V_eff + lambda0 S)^{-1} V_eff) = edf0``.
Fixing the effective degrees of freedom across pairs makes raw ``T`` values
comparable regardless of each pair's basis size or penalty scaling — at a
COMMON budget; across different budgets compare the normalized ``z`` the
ladder scan reports, never raw ``T``.

Ranking-only: calibration is by confirmatory refit, never by this number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

_EDF_TOL = 1e-6
_MAX_BISECT = 200


@dataclass(frozen=True)
class ScreenedPair:
    """Ranking output for one candidate pair."""

    statistic: float
    edf0: float
    lambda0: float


def _solve_psd(A: NDArray, B: NDArray) -> NDArray:
    """Solve A X = B for symmetric PSD A, falling back to a pseudo-inverse."""
    try:
        factor = scipy.linalg.cho_factor(A, check_finite=False)
        return scipy.linalg.cho_solve(factor, B, check_finite=False)
    except scipy.linalg.LinAlgError:
        return np.linalg.pinv(A, hermitian=True) @ B


def _edf(V: NDArray, S: NDArray, lam: float) -> float:
    return float(np.trace(_solve_psd(V + lam * S, V)))


def penalized_score_statistic(
    U: NDArray,
    V: NDArray,
    C: NDArray | None = None,
    M: NDArray | None = None,
    S_ti: NDArray | None = None,
    *,
    edf0: float = 4.0,
    U_nuisance: NDArray | None = None,
) -> ScreenedPair:
    """Rank one candidate pair by its penalized efficient-score statistic.

    ``C`` (overlap x tensor cross-moments) and ``M`` (overlap curvature)
    profile out the span the mains model already explains; ``U_nuisance`` is
    the overlap block's own score (zero at an exactly stationary fit, the
    penalty gradient ``S_M beta`` otherwise).  With ``S_ti`` absent or zero
    the statistic reduces to the unpenalized ``U' V^{-1} U`` and ``lambda0``
    is reported as 0.
    """
    U = np.asarray(U, dtype=np.float64)
    V = 0.5 * (np.asarray(V, dtype=np.float64) + np.asarray(V, dtype=np.float64).T)

    if (C is None) != (M is None):
        raise ValueError("C and M profile the overlap together; supply both or neither")
    if C is not None:
        C = np.asarray(C, dtype=np.float64)
        MinvC = _solve_psd(np.asarray(M, dtype=np.float64), C)
        V = V - C.T @ MinvC
        V = 0.5 * (V + V.T)
        if U_nuisance is not None:
            U = U - MinvC.T @ np.asarray(U_nuisance, dtype=np.float64)

    if S_ti is None or not np.any(S_ti):
        T = float(U @ _solve_psd(V, U))
        return ScreenedPair(statistic=T, edf0=_edf(V, np.zeros_like(V), 0.0), lambda0=0.0)

    S = 0.5 * (np.asarray(S_ti, dtype=np.float64) + np.asarray(S_ti, dtype=np.float64).T)

    # Bracket the EDF target. edf(lambda) decreases from rank(V) toward the
    # dimension of the penalty null space; clamp to the bracket edge when the
    # target is unreachable rather than failing the whole pair.
    scale = max(float(np.trace(V)), 1e-300) / max(float(np.trace(S)), 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale
    edf_lo, edf_hi = _edf(V, S, lo), _edf(V, S, hi)
    if edf0 >= edf_lo:
        lam = lo
    elif edf0 <= edf_hi:
        lam = hi
    else:
        for _ in range(_MAX_BISECT):
            lam = np.sqrt(lo * hi)
            achieved = _edf(V, S, lam)
            if abs(achieved - edf0) <= _EDF_TOL:
                break
            if achieved > edf0:
                lo = lam
            else:
                hi = lam

    achieved = _edf(V, S, lam)
    T = float(U @ _solve_psd(V + lam * S, U))
    return ScreenedPair(statistic=T, edf0=achieved, lambda0=float(lam))
