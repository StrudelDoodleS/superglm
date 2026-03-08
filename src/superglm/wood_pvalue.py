"""Wood (2013) smooth-term significance tests.

This module implements the Bayesian smooth-component test described in:

    Wood, S.N. (2013). On p-values for smooth components of an extended
    generalized additive model. Biometrika, 100(1), 221-228.

The implementation is written from the paper-level construction:
- move the term into a numerically stable test space via pivoted QR
- eigendecompose the term covariance in that space
- build a rank-reduced quadratic form using the alternative EDF ``edf1``
- evaluate the tail under a weighted chi-square mixture (known scale) or
  an F-style correction (estimated scale)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from superglm.davies import psum_chisq, satterthwaite


def _ordered_eigensystem(matrix: NDArray) -> tuple[NDArray, NDArray]:
    """Symmetric eigendecomposition with descending eigenvalues."""
    evals, evecs = np.linalg.eigh((matrix + matrix.T) / 2.0)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    if evals.size:
        # Eigenvectors are sign-indeterminate; fix the sign for stability.
        signs = np.sign(evecs[0])
        signs[signs == 0.0] = 1.0
        evecs = evecs * signs

    return evals, evecs


def _pivoted_test_space(X_j: NDArray, V_b_j: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Project a smooth into the orthogonal test space used by Wood's test."""
    from scipy.linalg import qr as scipy_qr

    _, r_factor, piv = scipy_qr(X_j, mode="economic", pivoting=True)
    cov_test = r_factor @ V_b_j[np.ix_(piv, piv)] @ r_factor.T
    return r_factor, cov_test, np.asarray(piv, dtype=np.int64)


def _clip_target_rank(edf1_j: float, p_g: int) -> float:
    """Clamp the target rank to the feasible coefficient dimension."""
    return max(0.0, min(float(edf1_j), float(p_g)))


def _effective_rank(evals: NDArray, requested_rank: float) -> tuple[int, float, int, float]:
    """Resolve the integer and fractional rank pieces against numerical rank."""
    if evals.size == 0 or evals[0] <= 0:
        return 0, 0.0, 0, 0.0

    rank_floor = max(0, int(math.floor(requested_rank)))
    frac = abs(requested_rank - rank_floor)
    working_dim = rank_floor + 1 if frac > 0.0 else rank_floor

    numerical_rank = int(np.sum(evals > evals[0] * np.finfo(float).eps ** 0.9))
    if numerical_rank < working_dim:
        working_dim = numerical_rank
        rank_floor = min(rank_floor, numerical_rank)
        frac = 0.0
        requested_rank = float(numerical_rank)

    return rank_floor, frac, working_dim, requested_rank


def _identity_rank_maps(
    evecs: NDArray,
    evals: NDArray,
    rank_floor: int,
) -> tuple[NDArray, NDArray, float]:
    """Maps for the integer-rank case."""
    if rank_floor <= 0:
        vec = evecs[:, :1] * math.sqrt(1.0 / evals[0])
        return vec, vec.copy(), 1.0

    scaled = evecs[:, :rank_floor] / np.sqrt(evals[:rank_floor])
    ref_df = 1.0 if rank_floor == 1 else float(rank_floor)
    return scaled, scaled.copy(), ref_df


def _fractional_rank_maps(
    evecs: NDArray,
    evals: NDArray,
    rank_floor: int,
    frac: float,
    working_dim: int,
    requested_rank: float,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """Maps for the fractional-rank interpolation in Wood (2013)."""
    if working_dim == 1:
        vec = evecs[:, :1] * math.sqrt(1.0 / evals[0])
        return vec, vec.copy(), np.array([1.0], dtype=np.float64), float(requested_rank)

    base = evecs[:, :working_dim].copy()
    if rank_floor > 1:
        base[:, : rank_floor - 1] /= np.sqrt(evals[: rank_floor - 1])

    coupling = math.sqrt(max(0.0, 0.5 * frac * (1.0 - frac)))
    boundary = np.array([[1.0, coupling], [coupling, frac]], dtype=np.float64)
    boundary /= np.sqrt(
        np.outer(evals[rank_floor - 1 : working_dim], evals[rank_floor - 1 : working_dim])
    )

    b_evals, b_evecs = np.linalg.eigh((boundary + boundary.T) / 2.0)
    b_evals = np.maximum(b_evals, 0.0)
    boundary_map = b_evecs @ np.diag(np.sqrt(b_evals)) @ b_evecs.T

    positive = base.copy()
    negative = base.copy()
    positive[:, rank_floor - 1 : working_dim] = (
        boundary_map @ base[:, rank_floor - 1 : working_dim].T
    ).T
    negative[:, rank_floor - 1 : working_dim] = (
        boundary_map @ np.diag([-1.0, 1.0]) @ base[:, rank_floor - 1 : working_dim].T
    ).T

    weights = np.ones(working_dim, dtype=np.float64)
    rp = frac + 1.0
    weights[rank_floor - 1] = (rp + math.sqrt(rp * (2.0 - rp))) / 2.0
    weights[-1] = rp - weights[rank_floor - 1]

    return positive, negative, weights, float(rank_floor + frac)


def _quadratic_stat(loadings: NDArray, projected_coef: NDArray) -> float:
    """Quadratic form induced by the test-space loadings."""
    return float(np.sum((loadings.T @ projected_coef) ** 2))


def _fractional_tail(
    stat: float,
    stat_alt: float,
    weights: NDArray,
    res_df: float,
) -> float:
    """Tail probability for the fractional-rank test statistic."""
    if res_df <= 0:
        p_primary, _ = psum_chisq(stat, weights)
        p_alt, _ = psum_chisq(stat_alt, weights)
        return 0.5 * (p_primary + p_alt)

    k0 = max(1, int(round(res_df)))
    base_df = np.ones(len(weights), dtype=np.float64)
    tail_df = np.concatenate([base_df, np.array([k0], dtype=np.float64)])

    mix_primary = np.concatenate([weights, np.array([-stat / k0], dtype=np.float64)])
    mix_alt = np.concatenate([weights, np.array([-stat_alt / k0], dtype=np.float64)])
    p_primary, _ = psum_chisq(0.0, mix_primary, df=tail_df)
    p_alt, _ = psum_chisq(0.0, mix_alt, df=tail_df)
    return 0.5 * (p_primary + p_alt)


def _fallback_tail(stat: float, stat_alt: float, ref_df: float, res_df: float) -> float:
    """Fallback when the mixture evaluation is unavailable or unstable."""
    if res_df <= 0:
        from scipy.stats import chi2 as chi2_dist

        return 0.5 * (chi2_dist.sf(stat, ref_df) + chi2_dist.sf(stat_alt, ref_df))

    from scipy.stats import f as f_dist

    return 0.5 * (
        f_dist.sf(stat / ref_df, ref_df, res_df) + f_dist.sf(stat_alt / ref_df, ref_df, res_df)
    )


def _mixture_pvalue(
    stat: float,
    weights: NDArray,
    res_df: float,
) -> float:
    """Reference tail for a weighted chi-square mixture."""
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if res_df > 0:
        from scipy.stats import f as f_dist

        mean = float(np.sum(weights))
        var = float(np.sum(2.0 * weights**2))
        if mean <= 0.0 or var <= 0.0:
            return 1.0 if stat <= 0.0 else 0.0

        scale = var / (2.0 * mean)
        df_num = 2.0 * mean**2 / var
        f_stat = stat / (scale * df_num)
        return float(f_dist.sf(f_stat, df_num, res_df))

    p_val, ifault = psum_chisq(stat, weights)
    if ifault != 0:
        p_val, _, _ = satterthwaite(stat, weights)
    return float(p_val)


def wood_test_smooth(
    beta_j: NDArray,
    X_j: NDArray,
    V_b_j: NDArray,
    edf1_j: float,
    res_df: float = -1.0,
) -> tuple[float, float, float]:
    """Smooth-term significance test from Wood (2013)."""
    p_g = len(beta_j)
    target_rank = _clip_target_rank(edf1_j, p_g)
    if target_rank < 1e-6:
        return 0.0, 1.0, 0.0

    r_factor, cov_test, piv = _pivoted_test_space(X_j, V_b_j)
    evals, evecs = _ordered_eigensystem(cov_test)
    if evals.size == 0 or evals[0] <= 0.0:
        return 0.0, 1.0, 0.0

    rank_floor, frac, working_dim, target_rank = _effective_rank(evals, target_rank)
    if working_dim == 0:
        return 0.0, 1.0, 0.0

    if frac > 0.0:
        pos_map, neg_map, weights, ref_df = _fractional_rank_maps(
            evecs, evals, rank_floor, frac, working_dim, target_rank
        )
    else:
        pos_map, neg_map, ref_df = _identity_rank_maps(
            evecs, evals, max(rank_floor, 0) if frac == 0.0 else 0
        )
        weights = None

    projected_coef = r_factor @ beta_j[piv]

    stat = _quadratic_stat(pos_map, projected_coef)
    stat_alt = _quadratic_stat(neg_map, projected_coef)

    if frac > 0.0:
        if weights is None:
            weights = np.array([1.0], dtype=np.float64)
            ref_df = 1.0
        p_val = _fractional_tail(stat, stat_alt, weights, res_df)
    else:
        p_val = 2.0

    if p_val > 1.0:
        p_val = _fallback_tail(stat, stat_alt, ref_df, res_df)

    return stat, float(min(1.0, p_val)), float(ref_df)
