"""PIRLS solver with pluggable penalty proximal operators."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.distributions import Distribution
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix, GroupMatrix
from superglm.links import Link
from superglm.penalties.base import Penalty
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
        eps = 1e-4 * L_g
        gram[np.diag_indices_from(gram)] += eps
        chol_groups.append(np.linalg.cholesky(gram))
    return L_groups, chol_groups


class _AndersonAccelerator:
    """Type-II Anderson acceleration for fixed-point iterations.

    Stores the last m iterates and residuals, then finds the linear
    combination that minimizes the residual norm.
    """

    def __init__(self, p: int, m: int = 5):
        self.m = m
        self._F_hist: list[NDArray] = []
        self._X_hist: list[NDArray] = []
        self._x_prev: NDArray | None = None

    def step(self, x_new: NDArray) -> NDArray | None:
        """Given x_{k+1} from BCD, return accelerated iterate or None."""
        if self._x_prev is not None:
            f = x_new - self._x_prev
            self._F_hist.append(f)
            self._X_hist.append(self._x_prev)
            if len(self._F_hist) > self.m:
                self._F_hist.pop(0)
                self._X_hist.pop(0)

        self._x_prev = x_new.copy()

        k = len(self._F_hist)
        if k < 2:
            return None

        F = np.column_stack(self._F_hist)
        gram = F.T @ F
        gram += np.eye(k) * 1e-10 * (np.trace(gram) + 1e-16)
        try:
            alpha = np.linalg.solve(gram, np.ones(k))
            alpha /= alpha.sum()
        except np.linalg.LinAlgError:
            return None

        X_arr = np.column_stack(self._X_hist)
        return X_arr @ alpha + F @ alpha


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
    anderson_memory: int = 0,
) -> PIRLSResult:
    """Single-pass PIRLS fit with proximal Newton BCD inner solver."""
    n, p = dm.shape
    beta = beta_init.copy() if beta_init is not None else np.zeros(p)

    # Initialize intercept
    if intercept_init is not None:
        intercept = intercept_init
    else:
        y_safe = np.where(y > 0, y, 0.1)
        intercept = float(link.link(np.atleast_1d(np.average(y_safe, weights=weights)))[0])

    gms = dm.group_matrices

    t_total = time.perf_counter()
    t_lipschitz_total = 0.0
    t_inner_total = 0.0
    total_inner_iters = 0

    dev_prev = np.inf
    for outer in range(max_iter_outer):
        t_outer_start = time.perf_counter()

        # Current predictions
        eta = dm.matvec(beta) + intercept + offset
        eta = np.clip(eta, -20, 20)
        mu = link.inverse(eta)

        # Working weights and response (PIRLS)
        V = family.variance(mu)
        V = np.maximum(V, 1e-10)
        dmu_deta = link.deriv_inverse(eta)
        W = weights * dmu_deta**2 / V
        z = eta + (y - mu) / dmu_deta

        # Per-group Hessians and Lipschitz constants
        t0 = time.perf_counter()
        L_groups, chol_groups = _compute_group_hessians(gms, W)
        t_lipschitz_total += time.perf_counter() - t0

        # Initialize residual
        r = z - dm.matvec(beta) - intercept - offset

        # Optional Anderson acceleration
        aa = _AndersonAccelerator(p, anderson_memory) if anderson_memory > 0 else None

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
            for gm, g, L_g, chol_g in zip(gms, groups, L_groups, chol_groups):
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

            # Optional Anderson acceleration
            if aa is not None:
                beta_acc = aa.step(beta)
                if beta_acc is not None:
                    for g, L_g in zip(groups, L_groups):
                        beta_acc[g.sl] = penalty.prox_group(
                            beta_acc[g.sl],
                            g,
                            1.0 / L_g,
                        )
                    beta = beta_acc
                    r = z - dm.matvec(beta) - intercept - offset

            # Check inner convergence
            change = np.max(np.abs(beta - beta_before))
            if change < tol * 0.01:
                break

        inner_iters = inner + 1
        total_inner_iters += inner_iters
        t_inner_total += time.perf_counter() - t_inner_start

        # Deviance for outer convergence
        eta_new = np.clip(dm.matvec(beta) + intercept + offset, -20, 20)
        mu_new = link.inverse(eta_new)
        dev = float(np.sum(weights * family.deviance_unit(y, mu_new)))

        t_outer_elapsed = time.perf_counter() - t_outer_start
        logger.info(
            f"  outer={outer + 1:3d}  bcd_cycles={inner_iters:4d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}  "
            f"time={t_outer_elapsed:.3f}s"
        )

        if abs(dev - dev_prev) / (abs(dev_prev) + 1.0) < tol:
            break
        dev_prev = dev

    t_elapsed = time.perf_counter() - t_total
    logger.info(
        f"  PIRLS done: {outer + 1} outer iters, {total_inner_iters} total BCD cycles, "
        f"{t_elapsed:.2f}s total"
    )
    logger.info(
        f"  Breakdown: group_lipschitz={t_lipschitz_total:.2f}s  bcd_cycles={t_inner_total:.2f}s"
    )

    # Effective df: Breheny & Huang (2009) formula for group lasso.
    # For active group g: df_g = p_g - (p_g - 1) * lambda1 * w_g / ||beta_g||
    # Accounts for within-group shrinkage. Reduces to 1 for size-1 groups (lasso).
    p_eff = 1.0  # intercept
    lam = penalty.lambda1 if penalty.lambda1 is not None else 0.0
    for g in groups:
        bg = beta[g.sl]
        norm_g = np.linalg.norm(bg)
        if norm_g > 1e-12:
            shrink = min(1.0, lam * g.weight / norm_g)
            p_eff += g.size - (g.size - 1) * shrink
    phi = dev / max(n - p_eff, 1)

    return PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=outer + 1,
        deviance=dev,
        converged=(outer + 1 < max_iter_outer),
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
    anderson_memory: int = 0,
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
        anderson_memory,
    )

    # Stage 2: if flavor, adjust weights and refit (warm-start both beta and intercept)
    if penalty.flavor is not None:
        adjusted_groups = penalty.flavor.adjust_weights(groups, result.beta)
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
            anderson_memory=anderson_memory,
        )

    return result
