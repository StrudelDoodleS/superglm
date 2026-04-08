"""Direct penalised IRLS solver (no BCD).

Solves the penalised GLM via iteratively reweighted least squares with a
single dense system solve per iteration:

    β = (X'WX + S)⁻¹ X'Wz

where p is ~50-80 (total model columns), making the p×p solve trivially
fast.  Uses gram-based operations (per-group gram + cross_gram) to form
X'WX without materialising the full (n, p) dense matrix.  For discretized
groups this reduces the O(n·p²) bottleneck to O(n_bins·K²) per group.

This replaces BCD when lambda1=0 (no L1/group lasso penalty), which is
the standard REML workflow where smoothing and optional term selection
are handled through the penalty structure. Without BCD, the 33-iteration aliasing from
shared B matrices between select=True subgroups vanishes entirely.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.linalg
import scipy.linalg.lapack
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, Distribution, clip_mu, initial_mean
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
    _block_xtwx_rhs,
)
from superglm.links import Link, stabilize_eta
from superglm.solvers.constrained_qp import solve_constrained_qp
from superglm.solvers.pirls import IterationDiagnostics, PIRLSResult
from superglm.solvers.scop_newton import scop_newton_step
from superglm.types import GroupSlice, PenaltyComponent

logger = logging.getLogger(__name__)


def _robust_solve(
    M: NDArray, rhs: NDArray, residual_tol: float = 1e-6
) -> tuple[NDArray, float, bool]:
    """Solve ``M @ x = rhs`` with pivoted Cholesky, SVD fallback.

    Uses a three-tier strategy (Higham, *Accuracy and Stability*, Ch. 10):

    1. **Pivoted Cholesky** (``dpstrf``): rank-revealing, handles PSD
       matrices.  If full-rank, solves via triangular back-substitution.
       If rank-deficient, solves the minimum-norm system from the leading
       *r* pivoted columns.
    2. **Residual check**: verifies ``||Mx - rhs|| / ||rhs|| < residual_tol``
       as a direct solve-quality signal.
    3. **SVD fallback**: backward-stable for any condition number.

    Parameters
    ----------
    M : (k, k) ndarray, symmetric positive (semi-)definite
    rhs : (k,) ndarray
    residual_tol : float
        Maximum acceptable relative residual from the solve.

    Returns
    -------
    x : (k,) ndarray
    cond_est : float
        Estimated condition number (from pivoted Cholesky diagonal or SVD).
    used_svd : bool
        True if the SVD path was taken.
    """
    k = M.shape[0]

    # === Primary path: Pivoted Cholesky (Higham Ch. 10.3) ===
    try:
        c, piv, rank, info = scipy.linalg.lapack.dpstrf(M, lower=0)
        if info < 0:
            raise np.linalg.LinAlgError("dpstrf: illegal argument")

        piv0 = piv - 1  # 0-indexed
        U = np.triu(c[:rank, :rank])  # rank x rank upper triangular

        # Permute rhs to pivoted order
        rhs_perm = rhs[piv0]

        if rank == k:
            # Full rank: solve U' U x_perm = rhs_perm
            y = scipy.linalg.solve_triangular(U, rhs_perm, lower=False, trans="T")
            x_perm = scipy.linalg.solve_triangular(U, y, lower=False)
            # Unpermute
            x = np.empty(k)
            x[piv0] = x_perm
        else:
            # Rank-deficient: solve in the leading rank subspace
            rhs_r = rhs_perm[:rank]
            # U[:rank, :rank] is the factor: solve U' U x_r = rhs_r
            y = scipy.linalg.solve_triangular(U, rhs_r, lower=False, trans="T")
            x_r = scipy.linalg.solve_triangular(U, y, lower=False)
            x_perm = np.zeros(k)
            x_perm[:rank] = x_r
            x = np.empty(k)
            x[piv0] = x_perm

        # Verify solution quality
        rhs_norm = np.linalg.norm(rhs)
        rel_residual = np.linalg.norm(M @ x - rhs) / max(rhs_norm, 1e-300)
        if rel_residual < residual_tol:
            diag_U = np.abs(np.diag(U))
            cond_est = float((diag_U.max() / max(diag_U.min(), 1e-300)) ** 2)
            return x, cond_est, False
    except (np.linalg.LinAlgError, ValueError):
        pass

    # === Fallback: Truncated SVD (backward-stable for any condition) ===
    U_svd, s, Vh = np.linalg.svd(M, full_matrices=False)
    thresh = s[0] * 1e-10
    mask = s > thresh
    inv_s = np.zeros_like(s)
    np.divide(1.0, s, out=inv_s, where=mask)
    x = (Vh.T * inv_s) @ (U_svd.T @ rhs)
    cond_est = float(s[0] / max(s[-1], 1e-300))
    return x, cond_est, True


def _build_penalty_matrix(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    p: int,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> NDArray:
    """Build block-diagonal penalty matrix S (p×p).

    If ``reml_penalties`` is provided, uses the multi-penalty path where
    multiple PenaltyComponents can accumulate (+=) into the same group
    block.  Each component carries its own omega and is keyed by a
    distinct lambda name.

    Otherwise falls back to the legacy single-penalty-per-group path.

    Parameters
    ----------
    reml_penalties : list of PenaltyComponent, optional
        When provided, each component contributes ``lam * omega_ssp``
        to the block identified by ``pc.group_sl``.  Multiple components
        sharing the same block are summed (e.g. tensor product marginals).
    """
    S = np.zeros((p, p))

    if reml_penalties is not None:
        for pc in reml_penalties:
            gm = group_matrices[pc.group_index]
            lam = lambda2[pc.name] if isinstance(lambda2, dict) else lambda2
            if lam == 0:
                continue
            omega_ssp = (
                pc.omega_ssp if pc.omega_ssp is not None else (gm.R_inv.T @ pc.omega_raw @ gm.R_inv)
            )
            S[pc.group_sl, pc.group_sl] += lam * omega_ssp

        # Also add SCOP penalties (not in reml_penalties).
        for gm, g in zip(group_matrices, groups):
            if g.scop_reparameterization is not None and g.penalized:
                lam_g = lambda2.get(g.name, 0.0) if isinstance(lambda2, dict) else lambda2
                if lam_g > 0:
                    S[g.sl, g.sl] += lam_g * g.scop_reparameterization.penalty_matrix()

        return S

    # Legacy single-penalty path (unchanged)
    for gm, g in zip(group_matrices, groups):
        if not g.penalized:
            continue

        if isinstance(lambda2, dict):
            lam_g = lambda2.get(g.name, 0.0)
        else:
            lam_g = lambda2

        if lam_g == 0:
            continue

        if isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
            omega = gm.omega
            if omega is None:
                continue
            S[g.sl, g.sl] = lam_g * gm.R_inv.T @ omega @ gm.R_inv
        elif g.scop_reparameterization is not None:
            # SCOP terms bypass SSP — penalty is already in solver coordinates.
            S_scop = g.scop_reparameterization.penalty_matrix()
            S[g.sl, g.sl] = lam_g * S_scop

    return S


def _sqrt_penalty_augmented(S: NDArray, p: int) -> NDArray:
    """Build (p+1, p+1) augmented sqrt-penalty for QR solver.

    Returns L_aug where L_aug.T @ L_aug has S in the [1:, 1:] block and
    zeros in the intercept row/column.
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 0.0)
    L_aug = np.zeros((p + 1, p + 1))
    L_aug[1:, 1:] = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    return L_aug


def _invert_xtwx_plus_penalty(
    XtWX: NDArray,
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> NDArray:
    """Invert ``X'WX + S(lambda2)`` for a fixed weighted Gram matrix.

    Parameters
    ----------
    S_override : (p, p) ndarray, optional
        Pre-built penalty matrix.  When provided, skips internal
        ``_build_penalty_matrix`` call entirely.
    reml_penalties : list of PenaltyComponent, optional
        Forwarded to ``_build_penalty_matrix`` for the multi-penalty path.
    """
    if S_override is not None:
        S = S_override
    else:
        p = XtWX.shape[0]
        S = _build_penalty_matrix(group_matrices, groups, lambda2, p, reml_penalties=reml_penalties)
    M_beta = XtWX + S
    H_inv, _, _ = _safe_decompose_H(M_beta)
    return H_inv


def _safe_decompose_H(H: NDArray, residual_tol: float = 1e-6) -> tuple[NDArray, float, bool]:
    """Decompose H = X'WX + S, returning its inverse and log-determinant.

    Uses a three-tier strategy (Higham, *Accuracy and Stability*, Ch. 10):

    1. **Pivoted Cholesky** (``dpstrf``): rank-revealing, O(p³).  Detects
       rank deficiency earlier and more cheaply than SVD.  For full-rank
       systems the inverse is computed via triangular solves.  For
       rank-deficient systems a truncated pseudo-inverse is formed from
       the leading *r* pivoted columns.
    2. **Residual spot-check**: after computing the inverse, verify
       ``||H @ H_inv[:, 0] - e_0|| < residual_tol``.  This is a direct
       solve-quality signal.  In practice, many rank-deficient cases
       still fail this check and fall through to SVD — the pivoted
       Cholesky win is earlier detection, not SVD elimination.
    3. **SVD fallback**: backward-stable for any condition number.

    Parameters
    ----------
    H : (p, p) ndarray
        The matrix to invert (typically X'WX + S).
    residual_tol : float
        Maximum acceptable residual from the inverse spot-check.

    Returns
    -------
    H_inv : (p, p) ndarray
        The inverse (or pseudo-inverse) of H.
    log_det_H : float
        log|H| (full-rank) or log|H|₊ (rank-deficient, positive eigenvalues).
    cholesky_ok : bool
        True if the pivoted-Cholesky path succeeded with acceptable quality.
    """
    p = H.shape[0]

    # === Primary path: Pivoted Cholesky (Higham Ch. 10.3) ===
    # dpstrf computes P' U' U P = H (upper) with column pivoting,
    # revealing the numerical rank.
    try:
        c, piv, rank, info = scipy.linalg.lapack.dpstrf(H, lower=0)
        piv0 = piv[:rank] - 1  # 0-indexed pivot indices for the leading rank columns

        if info < 0:
            raise np.linalg.LinAlgError("dpstrf: illegal argument")

        U = np.triu(c[:rank, :rank])  # rank x rank upper triangular factor

        if rank == p:
            # Full rank — invert via triangular solve on the permuted system.
            # P' U' U P = H  =>  H_inv = P' (U' U)^{-1} P
            U_inv = scipy.linalg.solve_triangular(U, np.eye(rank), lower=False)
            H_inv_perm = U_inv @ U_inv.T

            # Unpermute: H_inv[i, j] = H_inv_perm[inv_piv[i], inv_piv[j]]
            inv_piv = np.argsort(piv0)
            H_inv = H_inv_perm[np.ix_(inv_piv, inv_piv)]

            # Spot-check: verify first column of the inverse (O(p²))
            e0 = np.zeros(p)
            e0[0] = 1.0
            residual = np.linalg.norm(H @ H_inv[:, 0] - e0)
            if residual < residual_tol:
                diag_U = np.abs(np.diag(U))
                log_det_H = 2.0 * float(np.sum(np.log(diag_U)))
                return H_inv, log_det_H, True
        else:
            # Rank-deficient — truncated pseudo-inverse from leading r columns.
            # Only the first `rank` pivoted columns contribute.
            # U is rank x rank, factor of the rank x rank leading minor of P H P'.
            # The remaining (p - rank) directions are in the null space.
            U_inv = scipy.linalg.solve_triangular(U, np.eye(rank), lower=False)
            # Pseudo-inverse in permuted space: zero out null-space directions
            H_inv_perm = np.zeros((p, p))
            H_inv_perm[:rank, :rank] = U_inv @ U_inv.T

            inv_piv = np.argsort(piv - 1)  # full piv, 0-indexed
            H_inv = H_inv_perm[np.ix_(inv_piv, inv_piv)]

            # Spot-check
            e0 = np.zeros(p)
            e0[0] = 1.0
            residual = np.linalg.norm(H @ H_inv[:, 0] - e0)
            if residual < residual_tol:
                diag_U = np.abs(np.diag(U))
                log_det_H = 2.0 * float(np.sum(np.log(diag_U)))
                return H_inv, log_det_H, True
    except (np.linalg.LinAlgError, ValueError):
        pass

    # === Fallback: Truncated SVD (backward-stable for any condition) ===
    U_svd, s, Vh = np.linalg.svd(H, full_matrices=False)
    thresh = s[0] * 1e-10
    mask = s > thresh
    inv_s = np.zeros_like(s)
    np.divide(1.0, s, out=inv_s, where=mask)
    H_inv = (Vh.T * inv_s) @ Vh  # symmetric: Vh.T @ diag(inv_s) @ Vh

    pos_s = s[s > thresh]
    log_det_H = float(np.sum(np.log(pos_s))) if pos_s.size > 0 else 0.0

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
    tol: float = 1e-8,
    return_xtwx: bool = False,
    profile: dict | None = None,
    cache_out: dict | None = None,
    record_diagnostics: bool = False,
    direct_solve: str = "auto",
    convergence: str = "deviance",
    S_override: NDArray | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
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
        Frequency weights / sample_weight.
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
    S_override : (p, p) ndarray, optional
        Pre-built penalty matrix.  When provided, skips internal
        ``_build_penalty_matrix`` call entirely.
    reml_penalties : list of PenaltyComponent, optional
        Forwarded to ``_build_penalty_matrix`` for the multi-penalty path.

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
    if S_override is not None:
        S = S_override
    else:
        S = _build_penalty_matrix(gms, groups, lambda2, p, reml_penalties=reml_penalties)

    # ── Constrained QP support (monotone splines) ──
    has_constraints = any(g.constraints is not None for g in groups)
    prev_active_set: list[int] | None = None

    # ── SCOP monotone engine support ──
    _has_scop = any(g.monotone_engine == "scop" for g in groups)
    # group_idx -> {beta_scop, beta_scop_prev, reparam, B_scop, S_scop}
    _scop_state: dict[int, dict] = {}
    if _has_scop:
        for gi, g in enumerate(groups):
            if g.monotone_engine == "scop":
                reparam = g.scop_reparameterization
                B_scop = gms[gi].toarray()  # (n, q_eff)
                S_scop = reparam.penalty_matrix()
                _scop_state[gi] = {
                    "reparam": reparam,
                    "B_scop": B_scop,
                    "S_scop": S_scop,
                    "beta_scop": None,  # initialized on first iteration
                    "beta_scop_prev": None,
                }
        # Build mask of non-SCOP column indices for the reduced system
        _non_scop_cols = []
        _non_scop_groups_idx = []
        for gi, g in enumerate(groups):
            if gi not in _scop_state:
                _non_scop_cols.extend(range(g.start, g.end))
                _non_scop_groups_idx.append(gi)
        _non_scop_cols = np.array(_non_scop_cols, dtype=int)
        _p_reduced = len(_non_scop_cols)
        # Build reduced penalty matrix for non-SCOP groups
        _S_reduced = S[np.ix_(_non_scop_cols, _non_scop_cols)]
        # Build mapping from reduced beta index to full beta index
        _reduced_to_full = _non_scop_cols
        # Build reduced group slices (re-indexed for reduced system)
        _reduced_groups: list[GroupSlice] = []
        _reduced_gms = []
        col_offset_r = 0
        for gi in _non_scop_groups_idx:
            g = groups[gi]
            sz = g.size
            _reduced_groups.append(
                GroupSlice(
                    name=g.name,
                    start=col_offset_r,
                    end=col_offset_r + sz,
                    weight=g.weight,
                    penalized=g.penalized,
                    feature_name=g.feature_name,
                    subgroup_type=g.subgroup_type,
                    constraints=g.constraints,
                    monotone_engine=g.monotone_engine,
                )
            )
            _reduced_gms.append(gms[gi])
            col_offset_r += sz

    # QR pre-computation: materialise full design matrix once
    # Constrained QP / SCOP requires Gram path — force it if constraints present
    _use_qr = direct_solve == "qr" and not has_constraints and not _has_scop
    if _use_qr:
        has_disc = any(isinstance(gm, DiscretizedSSPGroupMatrix) for gm in gms)
        if has_disc:
            logger.warning(
                "direct_solve='qr' with discretized groups materialises the full "
                "(n, p) design matrix, defeating the O(n_bins) discretization "
                "benefit.  Consider direct_solve='auto' for large-n discrete fits."
            )
        _X_full = np.hstack([gm.toarray() for gm in gms])  # (n, p)
        _X_qr_aug = np.hstack([np.ones((n, 1)), _X_full])  # (n, p+1)
        _L_aug = _sqrt_penalty_augmented(S, p)  # (p+1, p+1)

    # tabmat acceleration: build SplitMatrix once for non-discrete paths.
    # R_inv is constant within a single fit_irls_direct call, so the
    # materialized X is valid for all IRLS iterations.
    _tabmat_split = dm.tabmat_split if not _use_qr else None

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
    _consecutive_svd = 0  # for auto-mode warning

    for it in range(max_iter):
        # Save previous solution for step halving
        beta_prev = beta.copy()
        intercept_prev = intercept

        # Working quantities from current eta/mu (already computed)
        _t0 = time.perf_counter()
        V = family.variance(mu)
        V = np.maximum(V, _VARIANCE_FLOOR)
        dmu_deta = link.deriv_inverse(eta)
        W = weights * dmu_deta**2 / V
        w_max = W.max()
        if w_max > 0:
            W = np.maximum(W, w_max * 1e-10)
        z = eta + (y - mu) / dmu_deta
        _t_working += time.perf_counter() - _t0

        if _use_qr:
            # QR path: solve via QR on [sqrt(W)·X_aug; sqrt(S_aug)].
            # No normal equations — backward-stable for any condition.
            _t0 = time.perf_counter()
            sqrtW = np.sqrt(W)
            z_off = z - offset
            A = np.vstack([sqrtW[:, None] * _X_qr_aug, _L_aug])
            rhs_qr = np.concatenate([sqrtW * z_off, np.zeros(p + 1)])
            Q, R = np.linalg.qr(A, mode="reduced")
            # Truncated SVD on R for near-singular truncation
            U_r, s_r, Vh_r = np.linalg.svd(R, full_matrices=False)
            thresh = s_r[0] * 1e-10
            inv_s = np.where(s_r > thresh, 1.0 / s_r, 0.0)
            beta_aug = (Vh_r.T * inv_s) @ (U_r.T @ (Q.T @ rhs_qr))
            _cond_est = float(s_r[0] / max(s_r[-1], 1e-300))
            # Report whether SVD truncation actually dropped any directions
            _used_svd = bool(np.any(s_r <= thresh))
            intercept = float(beta_aug[0])
            beta = beta_aug[1:]
            _t_solve += time.perf_counter() - _t0
        else:
            # Gram path: form X'WX via per-group gram, solve (p+1)×(p+1).
            _t0 = time.perf_counter()
            z_off = z - offset

            if _has_scop:
                # ── SCOP block-coordinate path ──────────────────────────
                # Step 1: Compute SCOP contribution to eta from current state
                eta_scop = np.zeros(n)
                for gi, st in _scop_state.items():
                    if st["beta_scop"] is None:
                        # Initialize SCOP beta on first iteration
                        g_i = groups[gi]
                        _lam_scop = (
                            lambda2.get(g_i.name, 0.0) if isinstance(lambda2, dict) else lambda2
                        )
                        st["beta_scop"] = st["reparam"].qp_initialize(
                            st["B_scop"],
                            z_off,
                            lambda_penalty=_lam_scop,
                            weights=W,
                        )
                    gamma_eff = st["reparam"].forward(st["beta_scop"])
                    eta_scop += st["B_scop"] @ gamma_eff

                # Step 2: Adjust working response by removing SCOP contribution
                z_adj = z_off - eta_scop
                Wz_adj = W * z_adj
                sum_W = float(np.sum(W))

                # Step 3: Build reduced Gram system (non-SCOP columns only)
                XtWX_r, XtW1_r, XtWz_r = _block_xtwx_rhs(
                    _reduced_gms,
                    _reduced_groups,
                    W,
                    Wz_adj,
                    tabmat_split=None,
                )

                # Build augmented (p_reduced+1, p_reduced+1) system
                M_aug_r = np.empty((_p_reduced + 1, _p_reduced + 1))
                M_aug_r[0, 0] = sum_W
                M_aug_r[0, 1:] = XtW1_r
                M_aug_r[1:, 0] = XtW1_r
                M_aug_r[1:, 1:] = XtWX_r + _S_reduced

                rhs_r = np.empty(_p_reduced + 1)
                rhs_r[0] = float(np.sum(Wz_adj))
                rhs_r[1:] = XtWz_r
                _t_gram += time.perf_counter() - _t0

                # Step 4: Solve for unconstrained coefficients
                _t0 = time.perf_counter()
                beta_aug_r, _cond_est, _used_svd = _robust_solve(M_aug_r, rhs_r)
                intercept = float(beta_aug_r[0])
                beta_reduced = beta_aug_r[1:]

                # Scatter reduced beta back into full beta vector
                beta = np.zeros(p)
                beta[_reduced_to_full] = beta_reduced
                _t_solve += time.perf_counter() - _t0

                # Step 5: Compute residual for SCOP Newton step
                eta_unconstrained = np.zeros(n)
                for gi_r, gi in enumerate(_non_scop_groups_idx):
                    g = groups[gi]
                    eta_unconstrained += gms[gi].matvec(beta[g.sl])
                eta_unconstrained += intercept

                # Step 6: Apply SCOP Newton step for each SCOP group
                for gi, st in _scop_state.items():
                    z_scop = z_off - eta_unconstrained
                    # Remove contributions from other SCOP groups
                    for gi2, st2 in _scop_state.items():
                        if gi2 != gi:
                            gamma2 = st2["reparam"].forward(st2["beta_scop"])
                            z_scop = z_scop - st2["B_scop"] @ gamma2

                    st["beta_scop_prev"] = st["beta_scop"].copy()
                    g_i = groups[gi]
                    _lam_scop = lambda2.get(g_i.name, 0.0) if isinstance(lambda2, dict) else lambda2
                    result = scop_newton_step(
                        B_scop=st["B_scop"],
                        W=W,
                        z=z_scop,
                        beta_scop=st["beta_scop"],
                        reparam=st["reparam"],
                        S_scop=st["S_scop"],
                        lambda2=_lam_scop,
                    )
                    st["beta_scop"] = result.beta_new

                # Step 7: Write gamma_eff (mapped coefficients) into full beta
                for gi, st in _scop_state.items():
                    g = groups[gi]
                    gamma_eff = st["reparam"].forward(st["beta_scop"])
                    beta[g.sl] = gamma_eff

            else:
                Wz = W * z_off
                sum_W = float(np.sum(W))

                # Combined gram + rmatvec: shares O(n) bincount for discretized groups
                XtWX, XtW1, XtWz = _block_xtwx_rhs(gms, groups, W, Wz, tabmat_split=_tabmat_split)

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

                # Solve — constrained QP or Cholesky with SVD fallback
                _t0 = time.perf_counter()
                if has_constraints:
                    # Assemble model-level constraint matrix from per-group constraints
                    A_blocks: list[np.ndarray] = []
                    b_blocks: list[np.ndarray] = []
                    for g in groups:
                        if g.constraints is not None:
                            A_model = np.zeros((g.constraints.n_constraints, p))
                            A_model[:, g.sl] = g.constraints.A
                            A_blocks.append(A_model)
                            b_blocks.append(g.constraints.b)

                    A_all = np.vstack(A_blocks)
                    b_all = np.concatenate(b_blocks)

                    # Profile out intercept (unconstrained):
                    # intercept = (rhs[0] - XtW1 @ beta) / sum_W
                    # Reduced system: H = XtWX + S, g_vec = XtWz - XtW1*(sum_Wz/sum_W)
                    H = M_aug[1:, 1:]  # XtWX + S
                    g_vec = rhs[1:] - rhs[0] * M_aug[0, 1:] / M_aug[0, 0]

                    qp_result = solve_constrained_qp(
                        H,
                        g_vec,
                        A_all,
                        b_all,
                        active_set_init=prev_active_set,
                    )
                    beta = qp_result.beta
                    intercept = float((rhs[0] - XtW1 @ beta) / sum_W)
                    prev_active_set = qp_result.active_set
                    _used_svd = False
                    _cond_est = 0.0
                else:
                    beta_aug, _cond_est, _used_svd = _robust_solve(M_aug, rhs)
                    intercept = float(beta_aug[0])
                    beta = beta_aug[1:]
                _t_solve += time.perf_counter() - _t0

            # Warning for auto mode: suggest QR after repeated SVD fallbacks
            if _used_svd:
                _consecutive_svd += 1
            else:
                _consecutive_svd = 0
            if direct_solve == "auto" and _consecutive_svd == 3:
                logger.warning(
                    "fit_irls_direct: %d consecutive SVD fallbacks (cond ~%.1e). "
                    "Consider direct_solve='qr' for near-collinear data.",
                    _consecutive_svd,
                    _cond_est,
                )

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
        # warrant halving.  The SVD fallback in _robust_solve() is the
        # primary defense against ill-conditioned overshoots.
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
                    f"  irls_direct iter={it + 1}: step halving {halving + 1}, dev={dev:.2e}"
                )
                if dev <= dev_prev:
                    break
            # Sync SCOP state after step halving: interpolate in working space
            if _has_scop and n_halvings > 0:
                frac = 0.5**n_halvings
                for gi, st in _scop_state.items():
                    if st["beta_scop_prev"] is not None:
                        st["beta_scop"] = (1 - frac) * st["beta_scop_prev"] + frac * st["beta_scop"]
                    # Re-write gamma_eff into beta for consistent eta
                    g = groups[gi]
                    beta[g.sl] = st["reparam"].forward(st["beta_scop"])

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
                    cond_estimate=_cond_est,
                    used_svd_fallback=_used_svd,
                )
            )

        logger.info(
            f"  irls_direct iter={it + 1:3d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}"
        )

        if not np.isfinite(dev):
            logger.warning(f"IRLS direct non-finite deviance at iter={it + 1}: dev={dev:.2e}")
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

    # QR path: compute gram quantities once at convergence for REML/edf/cache.
    if _use_qr or _has_scop:
        # QR path and SCOP path need a full-system Gram for edf / caching.
        z_off = z - offset
        Wz = W * z_off
        sum_W = float(np.sum(W))
        XtWX, XtW1, XtWz = _block_xtwx_rhs(gms, groups, W, Wz, tabmat_split=_tabmat_split)

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
    H_inv, log_det_H, _ = _safe_decompose_H(M_beta)
    XtWX_S_inv_beta = H_inv

    # Exact effective df: 1 (intercept) + trace((X'WX + S)^{-1} X'WX)
    F = XtWX_S_inv_beta @ XtWX_beta
    p_eff = 1.0 + float(np.trace(F))
    if profile is not None:
        _t_finalize = time.perf_counter() - _t0
        profile["irls_finalize_s"] = profile.get("irls_finalize_s", 0.0) + _t_finalize

    # Pearson-based phi for estimated-scale families (Tweedie, Gamma, NB2).
    # SuperGLM's sample_weight follows the prior-weight convention, so the
    # residual d.f. correction is observation-count based (n - edf), while
    # the weights still scale the Pearson numerator.
    V_final = np.maximum(family.variance(mu), _VARIANCE_FLOOR)
    pearson_chi2 = float(np.sum(weights * (y - mu) ** 2 / V_final))
    df_resid = max(float(len(y)) - p_eff, 1)
    phi = pearson_chi2 / df_resid

    result = PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=it + 1,
        deviance=dev,
        converged=converged,
        phi=phi,
        effective_df=p_eff,
        iteration_log=iteration_log if record_diagnostics else None,
        log_det_H=log_det_H,
    )

    if return_xtwx:
        return result, XtWX_S_inv_beta, XtWX_beta

    return result, XtWX_S_inv_beta
