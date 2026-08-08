"""REML/LAML objective function.

Laplace-approximate restricted maximum likelihood objective
(Wood 2011, Section 2, Eqs 4-5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, Gamma, Gaussian, Poisson, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import stabilize_eta
from superglm.reml.penalty_algebra import (
    build_penalty_matrix,
    cached_logdet_s_plus,
    compute_logdet_s_plus,
    compute_penalty_nullity,
    total_penalty_quadratic,
)
from superglm.reml.scale import (
    GammaScaleProfileData,
    ProfiledScaleTerm,
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
)
from superglm.reml.scop_geometry import decompose_on_scop_resolved_range
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import decompose_gram
from superglm.solvers.structured import SymmetricBlockOperator
from superglm.types import GroupSlice, PenaltyComponent

@dataclass(frozen=True)
class REMLObjectiveEvaluation:
    """One REML candidate evaluation and its reusable profiled-scale state."""

    value: float
    profiled_scale: ProfiledScaleTerm | None
    penalty_quad: float
    penalty_nullity: float | None
    penalized_deviance: float


def reml_laml_objective(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    y: NDArray,
    result: PIRLSResult,
    lambdas: dict[str, float],
    sample_weight: NDArray,
    offset_arr: NDArray,
    XtWX: NDArray | SymmetricBlockOperator | None = None,
    XtW1: NDArray | None = None,
    sum_W: float | None = None,
    penalty_caches: dict | None = None,
    log_det_H: float | None = None,
    hessian_rank: int | None = None,
    S_override: NDArray | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    scop_states: dict[int, dict] | None = None,
    tensor_pair_evaluations: dict | None = None,
    likelihood_size: float | None = None,
    gamma_scale_data: GammaScaleProfileData | None = None,
    return_evaluation: bool = False,
) -> float | REMLObjectiveEvaluation:
    """Laplace REML/LAML objective up to additive constants.

    Wood (2011) Section 2, Eqs (4)-(5): V(rho) = -l(beta_hat) + 0.5*beta_hat'S*beta_hat +
    0.5*log|H| - 0.5*log|S|_+. Known-scale: nll + 0.5*(penalty_quad + logdet_m -
    logdet_s). Estimated-scale: phi profiled out -> 0.5*(n-Mp)*log(D+beta_hat'S*beta_hat)
    replaces the nll + 0.5*penalty_quad terms.

    Parameters
    ----------
    log_det_H : float, optional
        Precomputed profiled-intercept determinant measure. At full rank this
        equals the ordinary log determinant of the augmented coefficient
        Hessian. Under rank truncation it is ``log(sum(W)) + log|H_c|_+`` on
        the retained centered slope coordinates; it is not the raw augmented
        matrix's Moore-Penrose pseudo-determinant.
    hessian_rank : int, optional
        Rank of the full observed coefficient Hessian associated with
        ``log_det_H``.  This must travel with an externally supplied
        determinant because Wood's ``M_p`` is counted only in the identified
        coefficient subspace.  When omitted, fitted Fisher/rank metadata is
        retained for compatibility.
    XtW1, sum_W : optional
        Cached intercept cross-products.  Together with ``XtWX`` these are
        sufficient to reconstruct the centered Schur complement when a
        precomputed full determinant is unavailable.

    Gaussian and Gamma use their family-specific Wood Eq. (4) scale
    profiles. Other estimated-scale families retain the historical reduced
    criterion until a family-specific scale profile is supplied.

    ``gamma_scale_data`` and ``likelihood_size`` let optimizers prepare
    fit-invariant row reductions once. Standalone callers may omit them; the
    objective then computes the required reduction for that evaluation.
    """
    mu = None
    retained_geometry_complete = (
        XtWX is None
        and log_det_H is not None
        and (S_override is not None or reml_penalties is not None)
    )
    if XtWX is None and not retained_geometry_complete:
        eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset_arr, link)
        mu = clip_mu(link.inverse(eta), distribution)
        V = distribution.variance(mu)
        dmu_deta = link.deriv_inverse(eta)
        W = sample_weight * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)
        moments = dm.execution_plan.moments(W, include_xtw=True)
        XtWX = moments.gram
        XtW1 = moments.xtw
        sum_W = float(np.sum(W, dtype=np.float64))

    if XtWX is not None and XtW1 is None and sum_W is None:
        if result.rank_info is not None:
            rank_mean = np.asarray(result.rank_info.mean_x, dtype=np.float64)
            if rank_mean.shape == (XtWX.shape[0],):
                sum_W = float(result.rank_info.sum_w)
                XtW1 = sum_W * rank_mean
        elif result.reml_geometry is not None and log_det_H is not None:
            # In-loop fits carry centered moments in the geometry summary
            # instead of rank metadata; the recovered values are identical.
            # Gated on log_det_H so callers that never supplied it (the
            # discrete path's cached-W results now also carry a summary) keep
            # the legacy slope-Gram determinant rather than silently flipping
            # to the intercept-profiled criterion.
            summary_mean = np.asarray(result.reml_geometry.mean_x, dtype=np.float64)
            if summary_mean.shape == (XtWX.shape[0],):
                sum_W = float(result.reml_geometry.sum_w)
                XtW1 = sum_W * summary_mean

    if (XtW1 is None) != (sum_W is None):
        raise ValueError("XtW1 and sum_W must be provided together")
    if XtW1 is not None:
        assert sum_W is not None
        XtW1 = np.asarray(XtW1, dtype=np.float64)
        if XtW1.shape != (XtWX.shape[0],):
            raise ValueError("XtW1 must match the slope coefficient dimension")
        if not np.isfinite(sum_W) or sum_W <= 0.0:
            raise ValueError("sum_W must be positive and finite")

    p = (
        np.asarray(S_override).shape[0]
        if S_override is not None
        else (XtWX.shape[0] if XtWX is not None else dm.p)
    )
    need_dense_penalty = (
        S_override is not None
        or scop_states is not None
        or reml_penalties is None
        or log_det_H is None
    )
    if S_override is not None:
        S: NDArray | None = np.asarray(S_override, dtype=np.float64)
    elif need_dense_penalty:
        S = build_penalty_matrix(
            dm.group_matrices, groups, lambdas, p, reml_penalties=reml_penalties
        )
    else:
        S = None
    if S is not None and S.shape != (p, p):
        raise ValueError("REML penalty must be square")
    if scop_states:
        from superglm.reml.scop_efs import compute_scop_aware_penalty_quad

        if S is None:  # pragma: no cover - dense SCOP invariant
            raise RuntimeError("SCOP objective is missing its dense penalty.")
        penalty_quad = compute_scop_aware_penalty_quad(
            result.beta,
            S,
            scop_states,
            lambdas,
            reml_penalties=reml_penalties,
        )
    elif reml_penalties is not None:
        penalty_quad = total_penalty_quadratic(
            result.beta,
            lambdas,
            reml_penalties,
            dm.group_matrices,
        )
    else:
        if S is None:  # pragma: no cover - fallback invariant
            raise RuntimeError("REML objective is missing its penalty geometry.")
        penalty_quad = float(result.beta @ S @ result.beta)

    # log|S|_+ -- use multi-penalty-aware path when reml_penalties available
    if reml_penalties is not None:
        logdet_s = compute_logdet_s_plus(
            lambdas,
            reml_penalties,
            tensor_pair_evaluations=tensor_pair_evaluations,
        )
    elif penalty_caches is not None:
        logdet_s = cached_logdet_s_plus(lambdas, penalty_caches)
    else:
        if S is None:  # pragma: no cover - fallback invariant
            raise RuntimeError("REML log-determinant fallback requires a dense penalty.")
        eigvals_s = np.linalg.eigvalsh(S)
        thresh_s = 1e-10 * max(eigvals_s.max(), 1e-12)
        pos_s = eigvals_s[eigvals_s > thresh_s]
        logdet_s = float(np.sum(np.log(pos_s))) if pos_s.size else 0.0

    centered_hessian_rank: int | None = None

    # The intercept-profiled Schur complement is
    # H_c = X'WX - X'W1 (X'W1)' / sum(W) + S.  The centering transform has
    # determinant one, hence at full rank
    # log|H_aug| = log(sum(W)) + log|H_c|. Under truncation the right-hand
    # side defines the translation-invariant identified-coordinate measure.
    if log_det_H is not None:
        logdet_m = log_det_H
    elif scop_states:
        from superglm.reml.scop_efs import assemble_joint_hessian

        if XtW1 is None or sum_W is None:
            raise ValueError("SCOP REML requires intercept cross-products")
        if S is None or not isinstance(XtWX, np.ndarray):
            raise ValueError("SCOP REML requires dense Gram and penalty matrices.")
        H_joint, _ = assemble_joint_hessian(
            XtWX + S,
            scop_states,
            XtW1=XtW1,
            sum_W=sum_W,
        )
        centered_decomposition = decompose_on_scop_resolved_range(H_joint, scop_states)
        centered_hessian_rank = centered_decomposition.rank
        logdet_m = float(np.log(sum_W) + centered_decomposition.log_pdet)
    else:
        if not isinstance(XtWX, np.ndarray):
            raise ValueError("A compact structured Gram requires a precomputed log_det_H.")
        if S is None:  # pragma: no cover - logdet fallback invariant
            raise RuntimeError("REML determinant fallback is missing its dense penalty.")
        if XtW1 is not None and sum_W is not None:
            centered_data_gram = XtWX - np.outer(XtW1, XtW1) / sum_W
            centered_hessian = centered_data_gram + S
            centered_decomposition = decompose_gram(centered_hessian)
            centered_hessian_rank = centered_decomposition.rank
            logdet_m = float(np.log(sum_W) + centered_decomposition.log_pdet)
        else:
            # Compatibility for callers that have cached only the historical
            # slope Gram.  Exact direct REML always supplies the authoritative
            # full determinant; new cached callers should provide XtW1/sum_W.
            M = XtWX + S
            decomposition = decompose_gram(M)
            centered_hessian_rank = decomposition.rank
            logdet_m = decomposition.log_pdet

    penalized_deviance = float(result.deviance + penalty_quad)

    # phi-profiled REML for estimated-scale families
    scale_known = getattr(distribution, "scale_known", True)
    if not scale_known:
        n = len(y)
        resolved_hessian_rank = hessian_rank
        if resolved_hessian_rank is None and centered_hessian_rank is not None:
            resolved_hessian_rank = 1 + centered_hessian_rank
        if resolved_hessian_rank is None:
            resolved_hessian_rank = result.reml_hessian_rank
        if resolved_hessian_rank is None and result.rank_info is not None:
            resolved_hessian_rank = 1 + result.rank_info.augmented.rank
        if resolved_hessian_rank is not None:
            M_p = compute_penalty_nullity(
                S,
                hessian_rank=resolved_hessian_rank,
                penalties=reml_penalties,
                lambdas=lambdas,
                coefficient_width=p,
            )
        else:
            # No centered/full-H metadata is available on this legacy path.
            # Its slope system was already rank-resolved above.
            M_p = compute_penalty_nullity(
                S,
                hessian_rank=1 + p,
                penalties=reml_penalties,
                lambdas=lambdas,
                coefficient_width=p,
            )
        profiled_scale: ProfiledScaleTerm | None = None
        if isinstance(distribution, Gaussian):
            if likelihood_size is None:
                likelihood_size = float(np.sum(sample_weight, dtype=np.float64))
            profiled_scale = profile_gaussian_reml_scale(
                penalized_deviance,
                likelihood_size,
                M_p,
            )
            scale_term = profiled_scale.criterion
        elif isinstance(distribution, Gamma):
            if gamma_scale_data is None:
                gamma_scale_data = prepare_gamma_reml_scale_data(y, sample_weight)
            profiled_scale = profile_gamma_reml_scale(
                gamma_scale_data,
                penalized_deviance,
                M_p,
            )
            scale_term = profiled_scale.criterion
        else:
            # Compatibility for estimated-scale families that do not yet have
            # an explicit Wood Eq. (4) profiler (notably Tweedie).
            d_plus_pq = max(penalized_deviance, 1e-300)
            scale_term = 0.5 * max(n - M_p, 1.0) * np.log(d_plus_pq)
        evaluation = REMLObjectiveEvaluation(
            value=float(0.5 * (logdet_m - logdet_s) + scale_term),
            profiled_scale=profiled_scale,
            penalty_quad=penalty_quad,
            penalty_nullity=M_p,
            penalized_deviance=penalized_deviance,
        )
        return evaluation if return_evaluation else evaluation.value

    if isinstance(distribution, Poisson) and (XtWX is not None or retained_geometry_complete):
        # Up to additive constants, Poisson negative log-likelihood is deviance / 2.
        nll = 0.5 * result.deviance
    else:
        if mu is None:
            eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset_arr, link)
            mu = clip_mu(link.inverse(eta), distribution)
        nll = -distribution.log_likelihood(y, mu, sample_weight, phi=1.0)
    evaluation = REMLObjectiveEvaluation(
        value=float(nll + 0.5 * (penalty_quad + logdet_m - logdet_s)),
        profiled_scale=None,
        penalty_quad=penalty_quad,
        penalty_nullity=None,
        penalized_deviance=penalized_deviance,
    )
    return evaluation if return_evaluation else evaluation.value


__all__ = ["REMLObjectiveEvaluation", "reml_laml_objective"]
