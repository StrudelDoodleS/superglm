"""Direct penalised IRLS solver (no BCD).

Solves the penalised GLM via iteratively reweighted least squares with a
single dense system solve per iteration:

    β = (X'WX + S)⁻¹ X'Wz

where p is ~50-80 (total model columns), making the p×p solve trivially
fast.  Uses gram-based operations (per-group gram + cross_gram) to form
X'WX without materialising the full (n, p) dense matrix.  For discretized
groups this reduces the O(n·p²) bottleneck to O(n_bins·K²) per group.

This replaces BCD when lambda1=0 (no L1/group lasso penalty), which is
the mgcv-style workflow where REML handles both smoothing and selection
through the double penalty.  Without BCD, the 33-iteration aliasing from
shared B matrices between select=True subgroups vanishes entirely.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.distributions import Distribution, clip_mu, initial_mean
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
    _block_xtwx_rhs,
)
from superglm.links import Link, stabilize_eta
from superglm.solvers.pirls import IterationDiagnostics, PIRLSResult
from superglm.types import GroupSlice

logger = logging.getLogger(__name__)


def _build_penalty_matrix(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    p: int,
) -> NDArray:
    """Build block-diagonal penalty matrix S (p×p).

    For each penalised SSP group: S[g.sl, g.sl] = lam_g * R_inv.T @ omega @ R_inv.
    Non-SSP and unpenalised groups contribute nothing.
    """
    S = np.zeros((p, p))
    for gm, g in zip(group_matrices, groups):
        if not g.penalized:
            continue
        if not isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
            continue
        omega = gm.omega
        if omega is None:
            continue

        if isinstance(lambda2, dict):
            lam_g = lambda2.get(g.name, 0.0)
        else:
            lam_g = lambda2

        S[g.sl, g.sl] = lam_g * gm.R_inv.T @ omega @ gm.R_inv

    return S


def _invert_xtwx_plus_penalty(
    XtWX: NDArray,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
) -> NDArray:
    """Invert ``X'WX + S(lambda2)`` for a fixed weighted Gram matrix."""
    p = XtWX.shape[0]
    S = _build_penalty_matrix(group_matrices, groups, lambda2, p)
    M_beta = XtWX + S
    H_inv, _, _ = _safe_decompose_H(M_beta)
    return H_inv


def _safe_decompose_H(H: NDArray) -> tuple[NDArray, float, bool]:
    """Decompose H = X'WX + S, returning its inverse and log-determinant.

    Attempts a fast, numerically stable Cholesky decomposition first.
    Falls back to a thresholded eigendecomposition if H is rank-deficient
    (e.g., due to collinear unpenalized categoricals or extreme IRLS weights).

    Parameters
    ----------
    H : (p, p) ndarray
        The matrix to invert (typically X'WX + S).

    Returns
    -------
    H_inv : (p, p) ndarray
        The inverse (or pseudo-inverse) of H.
    log_det_H : float
        log|H| (from Cholesky diagonal) or log|H|₊ (from positive eigenvalues).
    cholesky_ok : bool
        True if the Cholesky path succeeded.
    """
    p = H.shape[0]

    # === Primary path: Fast Cholesky ===
    try:
        L = scipy.linalg.cholesky(H, lower=True, check_finite=False)
        log_det_H = 2.0 * float(np.sum(np.log(np.diag(L))))
        H_inv = scipy.linalg.cho_solve((L, True), np.eye(p))
        return H_inv, log_det_H, True
    except np.linalg.LinAlgError:
        pass

    # === Fallback: Thresholded Eigendecomposition ===
    eigvals, eigvecs = np.linalg.eigh(H)
    threshold = 1e-6 * max(eigvals.max(), 1e-12)
    with np.errstate(divide="ignore"):
        inv_eigvals = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)
    H_inv = (eigvecs * inv_eigvals[None, :]) @ eigvecs.T

    pos_eigvals = eigvals[eigvals > threshold]
    log_det_H = float(np.sum(np.log(pos_eigvals))) if pos_eigvals.size > 0 else 0.0

    return H_inv, log_det_H, False


def fit_irls_direct(
    X: NDArray | DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    offset: NDArray | None = None,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_xtwx: bool = False,
    profile: dict | None = None,
    cache_out: dict | None = None,
    record_diagnostics: bool = False,
) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
    """Fit a penalised GLM via direct IRLS (no BCD).

    Solves β = (X'WX + S)⁻¹ X'Wz at each iteration.  Uses gram-based
    operations to form X'WX without materialising the full (n, p) dense
    matrix.  For discretized groups (DiscretizedSSPGroupMatrix), this
    reduces the per-iteration cost from O(n·p²) to O(n_bins·K²).

    Returns (PIRLSResult, XtWX_S_inv) where XtWX_S_inv is the (p, p)
    inverse from the final iteration, reusable for REML trace terms.

    Parameters
    ----------
    X : DesignMatrix or ndarray
        Design matrix (per-group or dense).
    y : (n,) array
        Response variable.
    weights : (n,) array
        Frequency weights / exposure.
    family : Distribution
        GLM family (Poisson, Gamma, NB2, etc.).
    link : Link
        Link function.
    groups : list of GroupSlice
        Group structure.
    lambda2 : float or dict
        Smoothing penalty weight(s).
    offset : (n,) array, optional
        Offset term.
    beta_init : (p,) array, optional
        Warm-start coefficients.
    intercept_init : float, optional
        Warm-start intercept.
    max_iter : int
        Maximum IRLS iterations (default 100).
    tol : float
        Deviance convergence tolerance (default 1e-6).
    return_xtwx : bool
        If True, also return the final weighted Gram matrix X'WX. Used by the
        REML outer loop to avoid rebuilding X'WX in cheap iterations when W is
        held fixed.
    record_diagnostics : bool
        If True, record per-iteration W/mu/eta stats on the result.

    Returns
    -------
    result : PIRLSResult
    XtWX_S_inv : (p, p) ndarray
        Inverse of (X'WX + S) from the final iteration.
    """
    if isinstance(X, DesignMatrix):
        dm = X
    else:
        from superglm.solvers.pirls import _wrap_dense_X

        dm = _wrap_dense_X(X, groups)

    n = dm.n
    p = dm.p
    gms = dm.group_matrices

    if offset is None:
        offset = np.zeros(n)

    beta = beta_init.copy() if beta_init is not None else np.zeros(p)

    if intercept_init is not None:
        intercept = intercept_init
    else:
        mu0 = initial_mean(y, weights, family)
        intercept = float(link.link(np.atleast_1d(mu0))[0])

    # Build penalty matrix S (p×p, block-diagonal)
    S = _build_penalty_matrix(gms, groups, lambda2, p)

    t_start = time.perf_counter()
    dev_prev = np.inf
    converged = False
    XtWX_beta = np.eye(p)  # will be overwritten

    # Phase timing accumulators
    _t_working = 0.0
    _t_gram = 0.0
    _t_solve = 0.0
    _t_deviance = 0.0

    # Pre-compute initial eta/mu once — reused as the first iteration's
    # working quantities, then updated at the end of each iteration.
    # This eliminates one redundant matvec + link.inverse per iteration.
    eta = stabilize_eta(dm.matvec(beta) + intercept + offset, link)
    mu = clip_mu(link.inverse(eta), family)
    iteration_log: list[IterationDiagnostics] = [] if record_diagnostics else []

    max_halving = 5  # max step-halving attempts per iteration
    for it in range(max_iter):
        # Save previous solution for step halving
        beta_prev = beta.copy()
        intercept_prev = intercept

        # Working quantities from current eta/mu (already computed)
        _t0 = time.perf_counter()
        V = family.variance(mu)
        V = np.maximum(V, 1e-10)
        dmu_deta = link.deriv_inverse(eta)
        W = weights * dmu_deta**2 / V
        z = eta + (y - mu) / dmu_deta
        _t_working += time.perf_counter() - _t0

        # Form augmented normal equations via gram-based operations.
        # M_aug = [[sum(W), X'W], [X'W, X'WX]] + S_aug
        # No full (n, p) matrix materialisation needed.
        _t0 = time.perf_counter()
        z_off = z - offset
        Wz = W * z_off
        sum_W = float(np.sum(W))

        # Combined gram + rmatvec: shares O(n) bincount for discretized groups
        XtWX, XtW1, XtWz = _block_xtwx_rhs(gms, groups, W, Wz)

        # Build augmented system (p+1, p+1)
        M_aug = np.empty((p + 1, p + 1))
        M_aug[0, 0] = sum_W
        M_aug[0, 1:] = XtW1
        M_aug[1:, 0] = XtW1
        M_aug[1:, 1:] = XtWX + S

        # RHS: X_aug' W (z - offset)
        rhs = np.empty(p + 1)
        rhs[0] = float(np.sum(Wz))
        rhs[1:] = XtWz
        _t_gram += time.perf_counter() - _t0

        # Solve augmented system — Cholesky (fast) with eigh fallback
        _t0 = time.perf_counter()
        try:
            L_aug = scipy.linalg.cholesky(M_aug, lower=True, check_finite=False)
            beta_aug = scipy.linalg.cho_solve((L_aug, True), rhs)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(M_aug)
            threshold = 1e-10 * max(eigvals.max(), 1e-12)
            with np.errstate(divide="ignore"):
                inv_eigvals = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)
            M_inv = (eigvecs * inv_eigvals[None, :]) @ eigvecs.T
            beta_aug = M_inv @ rhs

        intercept = float(beta_aug[0])
        beta = beta_aug[1:]
        _t_solve += time.perf_counter() - _t0

        # Update eta/mu from new beta — reused as next iteration's working
        # quantities (no redundant matvec at the start of the loop).
        _t0 = time.perf_counter()
        eta = stabilize_eta(dm.matvec(beta) + intercept + offset, link)
        mu = clip_mu(link.inverse(eta), family)
        dev = float(np.sum(weights * family.deviance_unit(y, mu)))
        _t_deviance += time.perf_counter() - _t0

        # Step halving: if deviance spiked dramatically (>2x), interpolate
        # between previous and current solution.  Small deviance increases
        # are normal in IRLS (especially non-canonical links) and don't
        # warrant halving.
        n_halvings = 0
        if np.isfinite(dev) and dev > 2.0 * dev_prev and np.isfinite(dev_prev):
            for halving in range(max_halving):
                beta = 0.5 * (beta + beta_prev)
                intercept = 0.5 * (intercept + intercept_prev)
                eta = stabilize_eta(dm.matvec(beta) + intercept + offset, link)
                mu = clip_mu(link.inverse(eta), family)
                dev_h = float(np.sum(weights * family.deviance_unit(y, mu)))
                if not np.isfinite(dev_h) or dev_h >= dev:
                    break
                n_halvings += 1
                dev = dev_h
                logger.info(
                    f"  irls_direct iter={it + 1}: step halving {halving + 1}, "
                    f"dev={dev:.2e}"
                )
                if dev <= dev_prev:
                    break

        # Record per-iteration diagnostics
        if record_diagnostics:
            w_ratio = W.max() / max(W.min(), 1e-300)
            k = min(5, n)
            top_idx = np.argpartition(W, -k)[-k:]
            bot_idx = np.argpartition(W, k)[:k]
            iteration_log.append(
                IterationDiagnostics(
                    iteration=it + 1,
                    deviance=dev,
                    w_min=float(W.min()),
                    w_max=float(W.max()),
                    w_ratio=w_ratio,
                    mu_min=float(mu.min()),
                    mu_max=float(mu.max()),
                    eta_min=float(eta.min()),
                    eta_max=float(eta.max()),
                    intercept=intercept,
                    step_halvings=n_halvings,
                    top_w_indices=top_idx[np.argsort(W[top_idx])[::-1]],
                    bottom_w_indices=bot_idx[np.argsort(W[bot_idx])],
                )
            )

        logger.info(
            f"  irls_direct iter={it + 1:3d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}"
        )

        if not np.isfinite(dev):
            logger.warning(
                f"IRLS direct non-finite deviance at iter={it + 1}: dev={dev:.2e}"
            )
            break

        if abs(dev - dev_prev) / (abs(dev_prev) + 1.0) < tol:
            converged = True
            break
        dev_prev = dev

    t_elapsed = time.perf_counter() - t_start
    logger.info(f"  IRLS direct done: {it + 1} iters, {t_elapsed:.2f}s")

    # Accumulate phase timing into the profile dict if provided
    if profile is not None:
        profile["irls_working_s"] = profile.get("irls_working_s", 0.0) + _t_working
        profile["irls_gram_s"] = profile.get("irls_gram_s", 0.0) + _t_gram
        profile["irls_solve_s"] = profile.get("irls_solve_s", 0.0) + _t_solve
        profile["irls_deviance_s"] = profile.get("irls_deviance_s", 0.0) + _t_deviance
        profile["irls_total_s"] = profile.get("irls_total_s", 0.0) + t_elapsed
        profile["irls_calls"] = profile.get("irls_calls", 0) + 1
        profile["irls_iters"] = profile.get("irls_iters", 0) + (it + 1)

    # Cache final-iteration RHS quantities for the cached-W fREML optimizer.
    # These allow re-solving the augmented system with a new penalty matrix S
    # without any data passes (O(p³) instead of O(n·K²) per group).
    if cache_out is not None:
        cache_out["XtWX"] = XtWX
        cache_out["XtWz"] = XtWz
        cache_out["XtW1"] = XtW1
        cache_out["sum_W"] = sum_W
        cache_out["sum_Wz"] = float(np.sum(Wz))

    # Compute (X'WX + S)^{-1} directly (NOT from augmented system, which gives
    # the Schur complement that accounts for intercept estimation — wrong for REML).
    # XtWX is already computed from the last iteration. Reuse S from above.
    _t0 = time.perf_counter()
    XtWX_beta = XtWX
    M_beta = XtWX_beta + S
    H_inv, _, _ = _safe_decompose_H(M_beta)
    XtWX_S_inv_beta = H_inv

    # Exact effective df: 1 (intercept) + trace((X'WX + S)^{-1} X'WX)
    F = XtWX_S_inv_beta @ XtWX_beta
    p_eff = 1.0 + float(np.trace(F))
    if profile is not None:
        _t_finalize = time.perf_counter() - _t0
        profile["irls_finalize_s"] = profile.get("irls_finalize_s", 0.0) + _t_finalize

    phi = dev / max(n - p_eff, 1)

    result = PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=it + 1,
        deviance=dev,
        converged=converged,
        phi=phi,
        effective_df=p_eff,
        iteration_log=iteration_log if record_diagnostics else None,
    )

    if return_xtwx:
        return result, XtWX_S_inv_beta, XtWX_beta

    return result, XtWX_S_inv_beta
