"""Wood (2011) Appendix B: stable log|Σ λ_i S_i|+ via similarity transform.

Standalone kernel for multi-penalty REML. Operates on lists of (q × q) penalty
matrices — no coupling to GroupInfo, DesignMatrix, or REML internals.

The algorithm recursively separates dominant penalty terms from subdominant
ones via Frobenius-norm scaling and eigendecomposition, producing a
block-diagonal structure with clean column separation for stable determinant
and derivative computation.

Reference: Wood, S.N. (2011), Appendix B, pp. 22-24.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_EPS = np.finfo(float).eps


@dataclass
class SimilarityTransformResult:
    """Result of Appendix B similarity transform.

    The transform separates the q-dimensional coefficient space into:
    - Q_plus: (q, rank) orthonormal basis for the penalized subspace
    - Q_zero: (q, q-rank) orthonormal basis for the null space of S
    - Q_full: (q, q) = [Q_plus | Q_zero], full orthogonal basis

    Q_plus is used for log|S|+, S⁺, E_sqrt, derivative formulas.
    Q_zero is needed for null-space handling and select=True style penalties.
    Q_full is the complete basis change when a full reparameterization is needed.
    """

    logdet_s_plus: float  # log|S|+ (positive eigenvalues only)
    S_pinv_plus: NDArray  # (q, q) pseudo-inverse on positive subspace
    Q_plus: NDArray  # (q, rank) penalized subspace basis
    Q_zero: NDArray  # (q, q-rank) null space basis
    E_sqrt: NDArray  # (q, q) stable square root: E'E ≈ S
    rank: int  # Numerical rank of S

    @property
    def Q_full(self) -> NDArray:
        """(q, q) full orthogonal basis [Q_plus | Q_zero]."""
        return np.hstack([self.Q_plus, self.Q_zero])


def similarity_transform_logdet(
    penalty_matrices: list[NDArray],
    lambdas: NDArray,
    eps_rank: float | None = None,
) -> SimilarityTransformResult:
    """Stable log|Σ λ_i S_i|+ via recursive similarity transform.

    Wood (2011) Appendix B, steps 1-9. Handles overlapping range spaces,
    wildly different λ magnitudes, and rank-deficient penalty matrices.

    Parameters
    ----------
    penalty_matrices : list of (q, q) ndarray
        Penalty matrices S_1, ..., S_M (symmetric PSD).
    lambdas : (M,) ndarray
        Smoothing parameters λ_1, ..., λ_M (positive).
    eps_rank : float, optional
        Threshold for rank determination. Default: eps^{1/3} ~ 6e-6.

    Returns
    -------
    SimilarityTransformResult
    """
    M = len(penalty_matrices)
    q = penalty_matrices[0].shape[0]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    if eps_rank is None:
        eps_rank = _EPS ** (1 / 3)

    # ── Initial Frobenius-norm transform ──────────────────────────
    # Ũ Λ̃ Ũ' = Σ S_i / ||S_i||_F
    frob_norms = np.array([np.linalg.norm(S, "fro") for S in penalty_matrices])
    frob_norms = np.maximum(frob_norms, 1e-300)  # avoid division by zero

    S_frob_sum = np.zeros((q, q))
    for i in range(M):
        S_frob_sum += penalty_matrices[i] / frob_norms[i]

    eigvals_init, U_init = np.linalg.eigh(S_frob_sum)
    # Keep only positive eigenvalues (reorder descending)
    idx = np.argsort(eigvals_init)[::-1]
    eigvals_init = eigvals_init[idx]
    U_init = U_init[:, idx]

    pos_mask = eigvals_init > eps_rank * max(eigvals_init.max(), 1e-12)
    n_pos = int(np.sum(pos_mask))

    if n_pos == 0:
        # Completely degenerate — all penalties in the null space
        return SimilarityTransformResult(
            logdet_s_plus=0.0,
            S_pinv_plus=np.zeros((q, q)),
            Q_plus=np.zeros((q, 0)),
            Q_zero=np.eye(q),
            E_sqrt=np.zeros((q, q)),
            rank=0,
        )

    # Transform to positive subspace: S̃_i = U+' S_i U+
    U_plus = U_init[:, :n_pos]
    Si_tilde = [U_plus.T @ S @ U_plus for S in penalty_matrices]

    # ── Recursive similarity transform (steps 1-9) ───────────────
    # Work entirely in the n_pos-dimensional positive subspace
    Q_dim = n_pos  # remaining dimension
    gamma = list(range(M))  # indices of remaining penalty terms
    K = 0  # columns already processed

    # Track transforms for the recursive block structure
    Q_inner = np.eye(Q_dim)  # accumulated inner transform

    while gamma and Q_dim > 0:
        # Step 1: Ω_i = ||S̃_i||_F · λ_i for remaining terms
        omegas = np.array([np.linalg.norm(Si_tilde[i], "fro") * lambdas[i] for i in gamma])

        if omegas.max() < 1e-300:
            break

        # Step 2: Separate dominant (α) from subdominant (γ')
        omega_max = omegas.max()
        alpha_mask = omegas >= eps_rank * omega_max
        alpha = [gamma[j] for j in range(len(gamma)) if alpha_mask[j]]
        gamma_prime = [gamma[j] for j in range(len(gamma)) if not alpha_mask[j]]

        # Step 3: Find rank r of Σ_{i∈α} S̃_i / ||S̃_i||_F
        S_alpha_frob = np.zeros((Q_dim, Q_dim))
        for i in alpha:
            fn = np.linalg.norm(Si_tilde[i], "fro")
            if fn > 1e-300:
                S_alpha_frob += Si_tilde[i] / fn

        eigvals_alpha, U_alpha = np.linalg.eigh(S_alpha_frob)
        idx_a = np.argsort(eigvals_alpha)[::-1]
        eigvals_alpha = eigvals_alpha[idx_a]
        U_alpha = U_alpha[:, idx_a]

        r = int(np.sum(eigvals_alpha > eps_rank * max(eigvals_alpha.max(), 1e-12)))

        # Step 4: If r == Q_dim, terminate (all remaining columns are dominant)
        if r == Q_dim:
            break

        if r == 0:
            # No dominant directions — remaining terms are all negligible
            break

        # Step 5: Eigendecompose Σ_{i∈α} λ_i S̃_i
        S_alpha_weighted = np.zeros((Q_dim, Q_dim))
        for i in alpha:
            S_alpha_weighted += lambdas[i] * Si_tilde[i]

        eigvals_w, U_w = np.linalg.eigh(S_alpha_weighted)
        idx_w = np.argsort(eigvals_w)[::-1]
        U_w = U_w[:, idx_w]

        U_r = U_w[:, :r]  # dominant eigenvectors
        U_n = U_w[:, r:]  # null/subdominant eigenvectors

        # Steps 6-7: Transform S̃_i
        # For dominant terms (α): project to full transform space
        # For subdominant terms (γ'): project to null space
        for i in alpha:
            # T_α' S̃_i T_α — concentrate into r×r block
            Si_tilde[i] = U_r.T @ Si_tilde[i] @ U_r

        for i in gamma_prime:
            # Project to null space of dominant terms
            Si_tilde[i] = U_n.T @ Si_tilde[i] @ U_n

        # Accumulate the inner transform
        Q_inner = Q_inner @ U_w

        # Step 9: Update bookkeeping
        K += r
        Q_dim -= r
        gamma = gamma_prime

    # ── Compute log|S|+ from the transformed structure ────────────
    # NOTE: this prototype forms S_transformed and runs eigvalsh on it.
    # A fully realized Appendix B would extract the determinant directly
    # from the block-diagonal structure built by the recursion. This is
    # correct but does not yet exploit the recursive column separation
    # for the determinant computation itself.
    S_transformed = np.zeros((n_pos, n_pos))
    for i in range(M):
        S_transformed += lambdas[i] * (
            Q_inner.T @ (U_plus.T @ penalty_matrices[i] @ U_plus) @ Q_inner
        )

    # Use eigenvalues of the transformed S for stable log-det
    eigvals_final = np.linalg.eigvalsh(S_transformed)
    pos_final = eigvals_final[eigvals_final > _EPS ** (2 / 3) * max(eigvals_final.max(), 1e-12)]
    rank = len(pos_final)

    if rank == 0:
        logdet = 0.0
    else:
        logdet = float(np.sum(np.log(pos_final)))

    # ── Build Q_plus and Q_zero ─────────────────────────────────────
    # Q_total maps from original q-space to the n_pos-dimensional positive
    # subspace. Within that subspace, the final eigendecomposition separates
    # penalized from near-null directions.
    Q_total = U_plus @ Q_inner  # (q, n_pos)

    # Split the transformed-space eigenvectors into penalized (V_plus)
    # and near-null (V_zero_internal)
    eigvals_t, eigvecs_t = np.linalg.eigh(S_transformed)
    idx_t = np.argsort(eigvals_t)[::-1]
    eigvals_t = eigvals_t[idx_t]
    eigvecs_t = eigvecs_t[:, idx_t]

    thresh_final = _EPS ** (2 / 3) * max(eigvals_t.max(), 1e-12)
    V_plus = eigvecs_t[:, :rank]  # (n_pos, rank) penalized directions
    V_zero_internal = eigvecs_t[:, rank:]  # (n_pos, n_pos-rank) internal null

    # Map back to original coordinates
    Q_plus = Q_total @ V_plus  # (q, rank) penalized subspace

    # Q_zero combines: (1) internal null from the positive subspace,
    # (2) the original common null space (U_init columns beyond n_pos)
    U_null_init = U_init[:, n_pos:]  # (q, q-n_pos) common null space
    Q_zero_internal = Q_total @ V_zero_internal  # (q, n_pos-rank)
    Q_zero = np.hstack([Q_zero_internal, U_null_init])  # (q, q-rank)

    # ── Pseudo-inverse on positive subspace ───────────────────────
    if rank > 0:
        inv_eigvals = np.zeros_like(eigvals_t)
        mask_t = eigvals_t > thresh_final
        np.divide(1.0, eigvals_t, out=inv_eigvals, where=mask_t)
        S_trans_pinv = (eigvecs_t * inv_eigvals) @ eigvecs_t.T
        S_pinv_plus = Q_total @ S_trans_pinv @ Q_total.T
    else:
        S_pinv_plus = np.zeros((q, q))

    # ── Stable square root E where E'E = S ────────────────────────
    # Preconditioning: P_ii = sqrt(|S'_ii|), LL' = P⁻¹ S P⁻¹, E = L' P
    if rank > 0:
        diag_S = np.abs(np.diag(S_transformed))
        P_diag = np.sqrt(np.maximum(diag_S, 1e-300))
        P_inv_diag = np.zeros_like(P_diag)
        np.divide(1.0, P_diag, out=P_inv_diag, where=P_diag > 1e-300)

        S_precond = (S_transformed * P_inv_diag[:, None]) * P_inv_diag[None, :]
        S_precond = 0.5 * (S_precond + S_precond.T)
        eigvals_pc = np.linalg.eigvalsh(S_precond)
        if eigvals_pc.min() < 0:
            S_precond += (abs(eigvals_pc.min()) + 1e-12) * np.eye(n_pos)

        try:
            L = np.linalg.cholesky(S_precond)
            E_inner = L.T * P_diag[None, :]
        except np.linalg.LinAlgError:
            eigvals_sq = np.maximum(eigvals_t, 0.0)
            E_inner = (eigvecs_t * np.sqrt(eigvals_sq)) @ eigvecs_t.T

        E_sqrt = Q_total @ E_inner @ Q_total.T
    else:
        E_sqrt = np.zeros((q, q))

    return SimilarityTransformResult(
        logdet_s_plus=logdet,
        S_pinv_plus=S_pinv_plus,
        Q_plus=Q_plus,
        Q_zero=Q_zero,
        E_sqrt=E_sqrt,
        rank=rank,
    )


def logdet_s_gradient(
    result: SimilarityTransformResult,
    penalty_matrices: list[NDArray],
    lambdas: NDArray,
) -> NDArray:
    """Gradient of log|S|+ w.r.t. ρ = log(λ).

    Wood (2011) Appendix B, Eq (16):
        ∂log|S|/∂ρ_j = λ_j tr(S⁻¹ S_j)

    Parameters
    ----------
    result : SimilarityTransformResult
        From ``similarity_transform_logdet``.
    penalty_matrices : list of (q, q) ndarray
    lambdas : (M,) ndarray

    Returns
    -------
    grad : (M,) ndarray
    """
    M = len(penalty_matrices)
    grad = np.zeros(M)
    S_pinv = result.S_pinv_plus

    for j in range(M):
        # tr(S⁻¹ S_j) = sum(S_pinv * S_j) via Frobenius inner product
        grad[j] = lambdas[j] * float(np.sum(S_pinv * penalty_matrices[j]))

    return grad


def logdet_s_hessian(
    result: SimilarityTransformResult,
    penalty_matrices: list[NDArray],
    lambdas: NDArray,
) -> NDArray:
    """Hessian of log|S|+ w.r.t. ρ = log(λ).

    Wood (2011) Appendix B, Eq (17):
        ∂²log|S|/(∂ρ_i ∂ρ_j) = δ^i_j λ_i tr(S⁻¹S_i)
                                 - λ_i λ_j tr(S⁻¹S_i S⁻¹S_j)

    Parameters
    ----------
    result : SimilarityTransformResult
    penalty_matrices : list of (q, q) ndarray
    lambdas : (M,) ndarray

    Returns
    -------
    hess : (M, M) ndarray
    """
    M = len(penalty_matrices)
    hess = np.zeros((M, M))
    S_pinv = result.S_pinv_plus

    # Precompute S⁻¹ S_j for each j — (q, q) each
    SinvSj = [S_pinv @ penalty_matrices[j] for j in range(M)]

    for i in range(M):
        for j in range(i, M):
            # -λ_i λ_j tr(S⁻¹S_i S⁻¹S_j)
            cross = -lambdas[i] * lambdas[j] * float(np.sum(SinvSj[i] * SinvSj[j].T))
            hess[i, j] = cross
            hess[j, i] = cross

        # Diagonal: δ^i_i λ_i tr(S⁻¹S_i)
        hess[i, i] += lambdas[i] * float(np.trace(SinvSj[i]))

    return hess
