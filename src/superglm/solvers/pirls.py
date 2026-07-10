"""PIRLS solver with pluggable penalty proximal operators."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, Distribution, initial_mean
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    GroupMatrix,
    _block_xtwx,
)
from superglm.links import Link
from superglm.penalties.base import Penalty, penalty_targets_group
from superglm.solvers.irls_state import _evaluate_irls_state, _IRLSState, _select_irls_trial
from superglm.types import GroupSlice

logger = logging.getLogger(__name__)


def _positive_working_weight_stats(W: NDArray) -> tuple[float, float, float]:
    """Return positive W minimum, maximum, and ratio, excluding zero-weight rows."""
    positive = W[W > 0]
    if positive.size == 0:
        return float("nan"), float("nan"), float("inf")

    positive_min = float(np.min(positive))
    positive_max = float(np.max(positive))
    if not np.isfinite(positive_max):
        ratio = float("inf")
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = float(np.divide(positive_max, positive_min))
    return positive_min, positive_max, ratio


@dataclass
class IterationDiagnostics:
    """Per-iteration IRLS diagnostics for debugging convergence issues."""

    iteration: int
    deviance: float
    w_min: float
    w_max: float
    w_ratio: float
    mu_min: float
    mu_max: float
    eta_min: float
    eta_max: float
    intercept: float
    step_halvings: int
    # Indices of the 5 observations with largest/smallest W
    top_w_indices: NDArray  # (5,) int
    bottom_w_indices: NDArray  # (5,) int
    # Condition estimate and SVD fallback flag (direct solver only)
    cond_estimate: float | None = None
    used_svd_fallback: bool | None = None
    raw_w_min: float | None = None
    raw_w_max: float | None = None
    raw_w_ratio: float | None = None
    eta_min_unclipped: float | None = None
    eta_max_unclipped: float | None = None
    eta_clipped: bool | None = None
    working_mu_min: float | None = None
    working_mu_max: float | None = None
    working_eta_min: float | None = None
    working_eta_max: float | None = None
    working_eta_min_unclipped: float | None = None
    working_eta_max_unclipped: float | None = None
    working_eta_clipped: bool | None = None
    step_rejected: bool = False


@dataclass
class PIRLSResult:
    beta: NDArray
    intercept: float
    n_iter: int
    deviance: float
    converged: bool
    phi: float
    effective_df: float
    iteration_log: list[IterationDiagnostics] | None = None
    log_det_H: float | None = None  # log|X'WX + S| from _safe_decompose_H  # noqa: N815


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
    record_diagnostics: bool = False,
    convergence: str = "deviance",
    S_override: NDArray | None = None,
) -> PIRLSResult:
    """Single-pass PIRLS fit with proximal Newton BCD inner solver."""
    n, p = dm.shape
    beta = beta_init.copy() if beta_init is not None else np.zeros(p)
    iteration_log: list[IterationDiagnostics] = [] if record_diagnostics else []

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

    # Freeze the fit-entry state so every trial is evaluated from fixed endpoints.
    committed = _evaluate_irls_state(dm, y, weights, family, link, offset, beta, intercept)
    dev_prev = committed.deviance
    if not np.isfinite(dev_prev):
        dev_prev = np.inf
    converged = False
    max_halving = 5  # max step-halving attempts per outer iteration
    for outer in range(max_iter_outer):
        t_outer_start = time.perf_counter()

        beta_prev = committed.beta
        intercept_prev = committed.intercept
        beta = committed.beta.copy()
        intercept = committed.intercept

        # Current predictions are the complete retained snapshot.
        eta_unclipped = committed.eta_unclipped
        eta = committed.eta
        mu = committed.mu

        # Working weights and response (PIRLS)
        V = family.variance(mu)
        V = np.maximum(V, _VARIANCE_FLOOR)
        dmu_deta = link.deriv_inverse(eta)
        W = weights * dmu_deta**2 / V
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
            delta_int: float = float(np.sum(W * r) / np.sum(W))
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
            change: float = float(np.max(np.abs(beta - beta_before)))
            if change < tol * 0.01:
                break

        inner_iters = inner + 1
        total_inner_iters += inner_iters
        t_inner_total += time.perf_counter() - t_inner_start

        proposal = _evaluate_irls_state(dm, y, weights, family, link, offset, beta, intercept)
        trial_cache: dict[float, _IRLSState] = {1.0: proposal}

        def evaluate_trial(alpha: float) -> _IRLSState:
            beta_trial = committed.beta + alpha * (proposal.beta - committed.beta)
            intercept_trial = committed.intercept + alpha * (
                proposal.intercept - committed.intercept
            )
            candidate = _evaluate_irls_state(
                dm,
                y,
                weights,
                family,
                link,
                offset,
                beta_trial,
                intercept_trial,
            )
            trial_cache[alpha] = candidate
            return candidate

        decision = _select_irls_trial(
            committed=committed,
            proposal=proposal,
            evaluate_state=evaluate_trial,
            max_halving=max_halving,
        )
        retained = committed if decision.step_rejected else trial_cache[decision.alpha]
        beta = retained.beta.copy()
        intercept = retained.intercept
        eta_new_unclipped = retained.eta_unclipped
        eta_new = retained.eta
        mu_new = retained.mu
        dev = retained.deviance
        n_halvings = decision.step_halvings
        step_rejected = decision.step_rejected
        if n_halvings:
            logger.info(
                "  PIRLS outer=%d: accepted step fraction %.5g after %d halvings, dev=%.2e",
                outer + 1,
                decision.alpha,
                n_halvings,
                dev,
            )

        # Warn on extreme working weight range (helps diagnose bad data)
        positive_w_min, positive_w_max, w_ratio = _positive_working_weight_stats(W)
        if w_ratio > 1e12:
            logger.warning(
                f"PIRLS outer={outer + 1}: extreme W ratio {w_ratio:.1e} "
                f"(positive W range [{positive_w_min:.2e}, {positive_w_max:.2e}])"
            )

        # Record per-iteration diagnostics
        if record_diagnostics:
            k = min(5, n)
            top_idx = np.argpartition(W, -k)[-k:]
            bot_idx = np.argpartition(W, k)[:k]
            working_eta_clipped = bool(
                float(np.min(eta_unclipped)) < float(np.min(eta))
                or float(np.max(eta_unclipped)) > float(np.max(eta))
            )
            eta_clipped = bool(
                float(np.min(eta_new_unclipped)) < float(np.min(eta_new))
                or float(np.max(eta_new_unclipped)) > float(np.max(eta_new))
            )
            iteration_log.append(
                IterationDiagnostics(
                    iteration=outer + 1,
                    deviance=dev,
                    w_min=float(W.min()),
                    w_max=float(W.max()),
                    w_ratio=w_ratio,
                    mu_min=float(mu_new.min()),
                    mu_max=float(mu_new.max()),
                    eta_min=float(eta_new.min()),
                    eta_max=float(eta_new.max()),
                    intercept=intercept,
                    step_halvings=n_halvings,
                    top_w_indices=top_idx[np.argsort(W[top_idx])[::-1]],
                    bottom_w_indices=bot_idx[np.argsort(W[bot_idx])],
                    raw_w_min=float(W.min()),
                    raw_w_max=float(W.max()),
                    raw_w_ratio=w_ratio,
                    eta_min_unclipped=float(np.min(eta_new_unclipped)),
                    eta_max_unclipped=float(np.max(eta_new_unclipped)),
                    eta_clipped=eta_clipped,
                    working_mu_min=float(mu.min()),
                    working_mu_max=float(mu.max()),
                    working_eta_min=float(eta.min()),
                    working_eta_max=float(eta.max()),
                    working_eta_min_unclipped=float(np.min(eta_unclipped)),
                    working_eta_max_unclipped=float(np.max(eta_unclipped)),
                    working_eta_clipped=working_eta_clipped,
                    step_rejected=step_rejected,
                )
            )

        t_outer_elapsed = time.perf_counter() - t_outer_start
        logger.info(
            f"  outer={outer + 1:3d}  bcd_cycles={inner_iters:4d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}  "
            f"time={t_outer_elapsed:.3f}s"
        )

        if step_rejected:
            logger.warning(
                "PIRLS rejected all trial steps at outer=%d; restored committed state",
                outer + 1,
            )
            break

        if not np.isfinite(dev):
            logger.warning(f"PIRLS non-finite deviance at outer={outer + 1}: dev={dev:.2e}")
            break

        if convergence == "coefficients":
            coef_change = float(np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta))))
            coef_change = max(
                coef_change,
                abs(intercept - intercept_prev) / max(1.0, abs(intercept)),
            )
            if coef_change < tol:
                converged = True
                break
        else:
            if abs(dev - dev_prev) / (abs(dev_prev) + 1.0) < tol:
                converged = True
                break
        committed = retained
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
        from superglm.reml.penalty_algebra import build_penalty_matrix

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
            if S_override is not None:
                # S_override is full (p x p) — slice to active columns
                active_idx: list[int] = []
                for ag in active_groups_edf:
                    # Map active group back to original group for slicing
                    orig_g = next(g for g in groups if g.name == ag.name)
                    active_idx.extend(range(orig_g.start, orig_g.end))
                active_idx = np.array(active_idx)
                S = S_override[np.ix_(active_idx, active_idx)]
            else:
                S = build_penalty_matrix(active_gms, active_groups_edf, lambda2, p_a)
            M = XtWX + S
            eigvals, eigvecs = np.linalg.eigh(M)
            # Keep the gram-path truncation aligned with the dense QR/SVD path:
            # singular-value cutoff ``rtol * s_max`` corresponds to
            # eigenvalue cutoff ``rtol**2 * eig_max``.
            threshold = (1e-6**2) * max(eigvals.max(), 1e-12)
            inv_eigvals = np.zeros_like(eigvals)
            np.divide(1.0, eigvals, out=inv_eigvals, where=eigvals > threshold)
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

    # Pearson-based phi for estimated-scale families (Tweedie, Gamma, NB2).
    # SuperGLM's sample_weight follows the prior-weight convention, so the
    # residual d.f. correction is observation-count based (n - edf), while
    # the weights still scale the Pearson numerator.
    V_final = np.maximum(family.variance(mu_new), _VARIANCE_FLOOR)
    pearson_chi2 = float(np.sum(weights * (y - mu_new) ** 2 / V_final))
    df_resid = max(float(len(y)) - p_eff, 1)
    phi = pearson_chi2 / df_resid

    return PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=outer + 1,
        deviance=dev,
        converged=converged,
        phi=phi,
        effective_df=p_eff,
        iteration_log=iteration_log if record_diagnostics else None,
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
    record_diagnostics: bool = False,
    convergence: str = "deviance",
    S_override: NDArray | None = None,
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
        record_diagnostics=record_diagnostics,
        convergence=convergence,
        S_override=S_override,
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
            record_diagnostics=record_diagnostics,
            convergence=convergence,
            S_override=S_override,
        )

    return result
