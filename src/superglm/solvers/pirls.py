"""PIRLS solver with pluggable penalty proximal operators."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.distributions import Distribution, clip_mu, initial_mean
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    GroupMatrix,
    _block_xtwx,
)
from superglm.links import Link, stabilize_eta
from superglm.penalties.base import Penalty, penalty_targets_group
from superglm.types import GroupSlice

logger = logging.getLogger(__name__)


@dataclass
class PIRLSResult:
    beta: NDArray
    intercept: float
    n_iter: int
    deviance: float
    converged: bool
    phi: float
    effective_df: float


def _compute_group_hessians(
    gms: list[GroupMatrix],
    W: NDArray,
) -> tuple[list[float], list[NDArray]]:
    """Per-group Lipschitz constants and regularised Cholesky factors.

    Returns (L_groups, chol_groups) where:
    - L_groups[g] = max eigenvalue of X_g' diag(W) X_g
    - chol_groups[g] = lower Cholesky factor of (H_g + eps*I)

    For typical group sizes (p_g <= 20) this is trivially cheap.
    Total cost is O(n * p) across all groups.
    """
    L_groups: list[float] = []
    chol_groups: list[NDArray] = []
    for gm in gms:
        gram = gm.gram(W)
        L_g = max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
        L_groups.append(L_g)
        # Regularise: SSP reparametrisation can leave near-singular or
        # numerically-negative-definite Hessians (eigenvalues ≈ -1e-13).
        # eps = 1e-4 * L_g keeps condition number ≤ 1e4, losing ≤ 4 digits.
        eps = max(1e-4 * L_g, 1e-8)
        gram[np.diag_indices_from(gram)] += eps
        chol_groups.append(np.linalg.cholesky(gram))
    return L_groups, chol_groups


def _fit_pirls_inner(
    dm: DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    penalty: Penalty,
    offset: NDArray,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter_outer: int = 100,
    max_iter_inner: int = 5,
    tol: float = 1e-6,
    active_set: bool = False,
    lambda2: float | dict[str, float] = 0.0,
) -> PIRLSResult:
    """Single-pass PIRLS fit with proximal Newton BCD inner solver."""
    n, p = dm.shape
    beta = beta_init.copy() if beta_init is not None else np.zeros(p)

    # Initialize intercept
    if intercept_init is not None:
        intercept = intercept_init
    else:
        mu0 = initial_mean(y, weights, family)
        intercept = float(link.link(np.atleast_1d(mu0))[0])

    gms = dm.group_matrices
    n_groups = len(groups)

    t_total = time.perf_counter()
    t_lipschitz_total = 0.0
    t_inner_total = 0.0
    total_inner_iters = 0
    total_groups_skipped = 0

    dev_prev = np.inf
    converged = False
    for outer in range(max_iter_outer):
        t_outer_start = time.perf_counter()

        # Current predictions
        eta = stabilize_eta(dm.matvec(beta) + intercept + offset, link)
        mu = clip_mu(link.inverse(eta), family)

        # Working weights and response (PIRLS)
        V = family.variance(mu)
        V = np.maximum(V, 1e-10)
        dmu_deta = link.deriv_inverse(eta)
        W = weights * dmu_deta**2 / V
        # Floor tiny W to prevent extreme condition numbers in Gram matrices.
        # Without this, observations near boundary predictions (e.g. mu→0 in
        # Poisson, mu→0/1 in Binomial) can produce W ratios > 1e15, causing
        # divergence.  The floor is relative to the max so it adapts to scale.
        w_max = W.max()
        if w_max > 0:
            W = np.maximum(W, w_max * 1e-12)
        z = eta + (y - mu) / dmu_deta

        # Per-group Hessians and Lipschitz constants
        t0 = time.perf_counter()
        L_groups, chol_groups = _compute_group_hessians(gms, W)
        t_lipschitz_total += time.perf_counter() - t0

        # Initialize residual
        r = z - dm.matvec(beta) - intercept - offset

        # Active set: track which groups can be skipped.
        # A group is inactive if beta_g == 0 AND ||grad_g|| < lambda1 * w_g
        # (KKT optimality for zeroed group).  First inner iter is always
        # a full sweep; subsequent iters skip inactive groups.
        group_active = [True] * n_groups

        # Inner loop: proximal Newton block coordinate descent
        t_inner_start = time.perf_counter()
        for inner in range(max_iter_inner):
            # Periodic residual refresh to avoid float drift
            if inner > 0 and inner % 5 == 0:
                r = z - dm.matvec(beta) - intercept - offset

            beta_before = beta.copy()

            # Update intercept (closed form, unpenalised)
            delta_int = np.sum(W * r) / np.sum(W)
            intercept += delta_int
            r -= delta_int

            # BCD cycle over groups (Newton step + prox)
            for gi, (gm, g, L_g, chol_g) in enumerate(zip(gms, groups, L_groups, chol_groups)):
                # Active set: skip groups confirmed inactive on previous sweep
                if active_set and inner > 0 and not group_active[gi]:
                    total_groups_skipped += 1
                    continue

                bg_old = beta[g.sl].copy()

                grad_g = -gm.rmatvec(W * r)
                # Newton direction via Cholesky solve: H_g^{-1} grad_g
                newton_dir = scipy.linalg.cho_solve(
                    (chol_g, True),
                    grad_g,
                )
                step_g = 1.0 / L_g
                bg_cand = bg_old - newton_dir
                bg_new = penalty.prox_group(bg_cand, g, step_g)

                d = bg_new - bg_old
                if np.any(d != 0):
                    r -= gm.matvec(d)
                    beta[g.sl] = bg_new

                # Active set: check KKT for zeroed groups after the update
                if active_set:
                    lam = penalty.lambda1 if penalty.lambda1 is not None else 0.0
                    if not penalty_targets_group(penalty, g):
                        lam = 0.0
                    if np.linalg.norm(bg_new) < 1e-12:
                        # Group is zero — check if gradient is below threshold
                        # Use the gradient *after* the update (recompute cheaply)
                        grad_after = -gm.rmatvec(W * r)
                        kkt_thr = lam * g.weight * 0.9  # safety margin
                        group_active[gi] = np.linalg.norm(grad_after) >= kkt_thr
                    else:
                        group_active[gi] = True

            # Check inner convergence
            change = np.max(np.abs(beta - beta_before))
            if change < tol * 0.01:
                break

        inner_iters = inner + 1
        total_inner_iters += inner_iters
        t_inner_total += time.perf_counter() - t_inner_start

        # Deviance for outer convergence
        eta_new = stabilize_eta(dm.matvec(beta) + intercept + offset, link)
        mu_new = clip_mu(link.inverse(eta_new), family)
        dev = float(np.sum(weights * family.deviance_unit(y, mu_new)))

        # Warn on extreme working weight range (helps diagnose bad data)
        w_ratio = W.max() / max(W.min(), 1e-300)
        if w_ratio > 1e12:
            logger.warning(
                f"PIRLS outer={outer + 1}: extreme W ratio {w_ratio:.1e} "
                f"(W range [{W.min():.2e}, {W.max():.2e}])"
            )

        t_outer_elapsed = time.perf_counter() - t_outer_start
        logger.info(
            f"  outer={outer + 1:3d}  bcd_cycles={inner_iters:4d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}  "
            f"time={t_outer_elapsed:.3f}s"
        )

        if not np.isfinite(dev):
            logger.warning(
                f"PIRLS non-finite deviance at outer={outer + 1}: dev={dev:.2e}"
            )
            break

        if dev > dev_prev * 10:
            # Deviance spike — log but continue; early IRLS iterations can
            # overshoot before settling, and aborting here prevents convergence
            # on datasets with extreme weight ranges.
            logger.info(
                f"  PIRLS outer={outer + 1}: deviance spike "
                f"(dev={dev:.2e}, prev={dev_prev:.2e}), continuing"
            )

        if abs(dev - dev_prev) / (abs(dev_prev) + 1.0) < tol:
            converged = True
            break
        dev_prev = dev

    t_elapsed = time.perf_counter() - t_total
    logger.info(
        f"  PIRLS done: {outer + 1} outer iters, {total_inner_iters} total BCD cycles, "
        f"{t_elapsed:.2f}s total"
    )
    extra = ""
    if active_set:
        total_group_updates = total_inner_iters * n_groups
        extra = f"  groups_skipped={total_groups_skipped}/{total_group_updates}"
    logger.info(
        f"  Breakdown: group_lipschitz={t_lipschitz_total:.2f}s  bcd_cycles={t_inner_total:.2f}s"
        + extra
    )

    # Effective df: exact hat-matrix trace when lambda2 > 0 (smoothing active),
    # Breheny-Huang (2009) group lasso formula when lambda2 = 0.
    has_smoothing = (isinstance(lambda2, dict) and any(v > 0 for v in lambda2.values())) or (
        not isinstance(lambda2, dict) and lambda2 > 0
    )

    if has_smoothing:
        # Exact: 1 + trace((X'WX + S)^{-1} X'WX) using final PIRLS working weights.
        from superglm.solvers.irls_direct import _build_penalty_matrix

        active_groups_edf: list[GroupSlice] = []
        active_gms: list[GroupMatrix] = []
        col = 0
        for gm, g in zip(gms, groups):
            if np.linalg.norm(beta[g.sl]) > 1e-12:
                p_g = gm.shape[1]
                active_groups_edf.append(
                    GroupSlice(
                        name=g.name,
                        start=col,
                        end=col + p_g,
                        weight=g.weight,
                        penalized=g.penalized,
                        feature_name=g.feature_name,
                        subgroup_type=g.subgroup_type,
                    )
                )
                active_gms.append(gm)
                col += p_g

        if active_gms:
            p_a = col
            XtWX = _block_xtwx(active_gms, active_groups_edf, W)
            S = _build_penalty_matrix(active_gms, active_groups_edf, lambda2, p_a)
            M = XtWX + S
            eigvals, eigvecs = np.linalg.eigh(M)
            threshold = 1e-6 * max(eigvals.max(), 1e-12)
            inv_eigvals = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)
            M_inv = (eigvecs * inv_eigvals[None, :]) @ eigvecs.T
            p_eff = 1.0 + float(np.trace(M_inv @ XtWX))
        else:
            p_eff = 1.0
    else:
        # Breheny & Huang (2009) formula for group lasso (no smoothing).
        # df_g = p_g - (p_g - 1) * lambda1 * w_g / ||beta_g||
        p_eff = 1.0  # intercept
        lam = penalty.lambda1 if penalty.lambda1 is not None else 0.0
        for g in groups:
            bg = beta[g.sl]
            norm_g = np.linalg.norm(bg)
            if norm_g > 1e-12:
                if not penalty_targets_group(penalty, g):
                    p_eff += g.size
                else:
                    shrink = min(1.0, lam * g.weight / norm_g)
                    p_eff += g.size - (g.size - 1) * shrink

    phi = dev / max(n - p_eff, 1)

    return PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=outer + 1,
        deviance=dev,
        converged=converged,
        phi=phi,
        effective_df=p_eff,
    )


def _wrap_dense_X(X: NDArray, groups: list[GroupSlice]) -> DesignMatrix:
    """Wrap a dense NDArray into a DesignMatrix for backward compatibility."""
    n, p = X.shape
    gms = [DenseGroupMatrix(X[:, g.sl]) for g in groups]
    return DesignMatrix(gms, n, p)


def fit_pirls(
    X: NDArray | DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    penalty: Penalty,
    offset: NDArray | None = None,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter_outer: int = 100,
    max_iter_inner: int = 5,
    tol: float = 1e-6,
    active_set: bool = False,
    lambda2: float | dict[str, float] = 0.0,
) -> PIRLSResult:
    """Fit a penalised GLM via PIRLS with proximal Newton BCD.

    If the penalty has a flavor (e.g. Adaptive), a two-stage fit is performed:
    1. Fit with uniform weights → beta_init
    2. Flavor adjusts group weights based on beta_init
    3. Refit with adjusted weights (warm started from stage 1)
    """
    if isinstance(X, DesignMatrix):
        dm = X
        n = dm.n
    else:
        dm = _wrap_dense_X(X, groups)
        n = X.shape[0]

    if offset is None:
        offset = np.zeros(n)

    # Stage 1: initial fit
    result = _fit_pirls_inner(
        dm,
        y,
        weights,
        family,
        link,
        groups,
        penalty,
        offset,
        beta_init,
        intercept_init,
        max_iter_outer,
        max_iter_inner,
        tol,
        active_set,
        lambda2=lambda2,
    )

    # Stage 2: if flavor, adjust weights and refit (warm-start both beta and intercept)
    if penalty.flavor is not None:
        adjusted_groups = penalty.flavor.adjust_weights(
            groups, result.beta, group_matrices=dm.group_matrices
        )
        result = _fit_pirls_inner(
            dm,
            y,
            weights,
            family,
            link,
            adjusted_groups,
            penalty,
            offset,
            beta_init=result.beta,
            intercept_init=result.intercept,
            max_iter_outer=max_iter_outer,
            max_iter_inner=max_iter_inner,
            tol=tol,
            active_set=active_set,
            lambda2=lambda2,
        )

    return result
