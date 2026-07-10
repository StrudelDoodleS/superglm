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
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import superglm.solvers.scop_exact_support as scop_exact_support
from superglm.distributions import _VARIANCE_FLOOR, Distribution, initial_mean
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    _block_xtwx_rhs,
)
from superglm.links import Link
from superglm.solvers.centered_system import (
    CenteredSystem,
    build_centered_system,
    grouped_augmented_factor,
    grouped_weighted_factor,
    refresh_centered_rhs,
)
from superglm.solvers.constrained_qp import solve_constrained_qp
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _IRLSState,
    _select_irls_trial,
)
from superglm.solvers.pirls import (
    IterationDiagnostics,
    PIRLSResult,
    _positive_working_weight_stats,
)
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankInfo,
    decompose_factor,
    decompose_gram,
    decompose_symmetric,
    needs_factor_certification,
)
from superglm.solvers.scop import SCOPSolverReparam
from superglm.solvers.scop_newton import scop_joint_newton_step, scop_newton_step
from superglm.types import GroupSlice, PenaltyComponent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SCOPGroupSpec:
    """Static definition of one SCOP group, separate from trial state."""

    group_index: int
    group: GroupSlice
    reparam: SCOPSolverReparam
    B_scop: NDArray
    S_scop: NDArray
    bin_idx: NDArray | None


@dataclass(frozen=True)
class _SCOPGroupState:
    """Dynamic SCOP state committed atomically with an IRLS snapshot."""

    group_index: int
    beta_eff: NDArray
    gamma_eff: NDArray
    H_scop_penalized: NDArray | None
    last_step_norm: float
    last_fisher_fallback: bool


@dataclass(frozen=True)
class _SCOPTrialState:
    """One complete mixed ordinary/SCOP trial."""

    irls: _IRLSState
    groups: tuple[_SCOPGroupState, ...]


def _evaluate_scop_trial(
    *,
    committed: _SCOPTrialState,
    proposed: _SCOPTrialState,
    alpha: float,
    specs: dict[int, _SCOPGroupSpec],
    dm: DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    offset: NDArray,
) -> _SCOPTrialState:
    """Evaluate a fixed-endpoint SCOP trial by interpolating latent state."""
    beta_trial = committed.irls.beta + alpha * (proposed.irls.beta - committed.irls.beta)
    intercept_trial = committed.irls.intercept + alpha * (
        proposed.irls.intercept - committed.irls.intercept
    )
    trial_groups: list[_SCOPGroupState] = []
    for committed_group, proposed_group in zip(committed.groups, proposed.groups, strict=True):
        if committed_group.group_index != proposed_group.group_index:
            raise ValueError("SCOP trial group ordering does not match")
        spec = specs[committed_group.group_index]
        beta_eff = committed_group.beta_eff + alpha * (
            proposed_group.beta_eff - committed_group.beta_eff
        )
        gamma_eff = spec.reparam.forward(beta_eff)
        beta_trial[spec.group.sl] = gamma_eff
        trial_groups.append(
            _SCOPGroupState(
                group_index=committed_group.group_index,
                beta_eff=_immutable_array(beta_eff),
                gamma_eff=_immutable_array(gamma_eff),
                H_scop_penalized=None,
                last_step_norm=float(np.linalg.norm(beta_eff - committed_group.beta_eff)),
                last_fisher_fallback=proposed_group.last_fisher_fallback,
            )
        )

    irls = _evaluate_irls_state(dm, y, weights, family, link, offset, beta_trial, intercept_trial)
    return _SCOPTrialState(irls=irls, groups=tuple(trial_groups))


def _has_constant_irls_weights(family: Distribution, link: Link) -> bool:
    """Return True when PIRLS weights are independent of ``mu``.

    The direct solver can reuse X'WX only when
    ``(dmu/deta)^2 / V(mu)`` is exactly constant.  Keep this deliberately
    conservative so performance never changes the fitted problem.
    """
    from superglm.distributions import Gamma, Gaussian
    from superglm.links import IdentityLink, LogLink

    return (isinstance(family, Gaussian) and isinstance(link, IdentityLink)) or (
        isinstance(family, Gamma) and isinstance(link, LogLink)
    )


def _robust_solve(
    M: NDArray, rhs: NDArray, residual_tol: float = 1e-6
) -> tuple[NDArray, float, bool]:
    """Compatibility wrapper around the shared equilibrated rank policy."""
    decomposition = decompose_gram(M, residual_tol=residual_tol)
    used_spectral_fallback = decomposition.method not in {
        "cholesky",
        "pivoted_cholesky",
    }
    return (
        decomposition.solve(rhs),
        decomposition.pre_truncation_condition,
        used_spectral_fallback,
    )


def _solve_profiled_intercept_from_h_inv(
    H_inv: NDArray,
    XtWz: NDArray,
    XtW1: NDArray,
    sum_W: float,
    sum_Wz: float,
) -> tuple[NDArray, float]:
    """Solve beta/intercept from H^{-1} with the intercept profiled out."""
    h_z = H_inv @ XtWz
    h_1 = H_inv @ XtW1
    denom = sum_W - float(XtW1 @ h_1)
    intercept = float((sum_Wz - XtW1 @ h_z) / denom)
    beta = h_z - h_1 * intercept
    return beta, intercept


def _build_penalty_matrix(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    p: int,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> NDArray:
    """Backward-compatible wrapper for the shared REML penalty builder."""
    from superglm.reml.penalty_algebra import build_penalty_matrix

    return build_penalty_matrix(
        group_matrices,
        groups,
        lambda2,
        p,
        reml_penalties=reml_penalties,
    )


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
    """Compatibility wrapper returning inverse, pseudo-logdet, and fast-path flag."""
    decomposition = decompose_symmetric(H, residual_tol=residual_tol)
    cholesky_ok = decomposition.method in {"cholesky", "pivoted_cholesky"}
    return decomposition.pseudo_inverse(), decomposition.log_pdet, cholesky_ok


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
    return_scop_state: bool = False,
    _scop_joint: bool = True,
    scop_state_init: dict[int, dict] | None = None,
    debug_recorder=None,
    debug_context: dict[str, object] | None = None,
    compute_rank_info: bool = True,
    _return_working_system: bool = False,
    _compute_fit_statistics: bool = True,
    _deviance_init: float | None = None,
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
    compute_rank_info : bool
        If False, skip data-subspace metadata used only by retained-fit
        inference. Intermediate REML fits still compute the coefficient and
        augmented decompositions needed for their objective and EDF.
    _return_working_system : bool
        Internal fREML performance-iteration mode. Return the centered system
        used for the last coefficient update instead of rebuilding it at the
        proposed coefficients. The authoritative final fit must leave this
        False so exported inference remains tied to the retained model.
    _compute_fit_statistics : bool
        Internal optimization switch. If False, omit EDF and scale summaries
        that the fREML outer loop does not consume. The authoritative final
        fit must leave this True.
    _deviance_init : float, optional
        Previously evaluated deviance at ``beta_init``/``intercept_init``.
        Used by private fREML steps to avoid repeating a full response scan.
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
    A_all: NDArray | None = None
    b_all: NDArray | None = None
    if has_constraints:
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

    # ── SCOP monotone engine support ──
    _has_scop = any(g.monotone_engine == "scop" for g in groups)
    if (not _compute_fit_statistics and compute_rank_info) or (
        _return_working_system and (compute_rank_info or has_constraints or _has_scop)
    ):
        raise ValueError("intermediate REML shortcuts require rank metadata to be disabled")
    _n_scop_groups = sum(g.monotone_engine == "scop" for g in groups)
    _expose_exact_support_state = False
    # group_idx -> {beta_scop, beta_scop_prev, reparam, B_scop, S_scop}
    _scop_state: dict[int, dict] = {}
    if _has_scop:
        for gi, g in enumerate(groups):
            if g.monotone_engine == "scop":
                cached_scop_state = scop_state_init.get(gi) if scop_state_init is not None else None
                reparam = g.scop_reparameterization
                cached_S_scop = (
                    None if cached_scop_state is None else cached_scop_state.get("S_scop")
                )
                if isinstance(cached_S_scop, np.ndarray) and cached_S_scop.shape == (
                    g.size,
                    g.size,
                ):
                    S_scop = cached_S_scop
                else:
                    S_scop = reparam.penalty_matrix()
                _gm = gms[gi]

                # Warm-start beta_scop from previous outer EFS iteration if available
                warm_beta_scop = None
                if cached_scop_state is not None:
                    prev = cached_scop_state["beta_eff"]
                    q_eff = S_scop.shape[0]
                    if prev.shape == (q_eff,):
                        warm_beta_scop = prev.copy()

                if isinstance(_gm, DiscretizedSCOPGroupMatrix):
                    state = {
                        "reparam": reparam,
                        "B_scop": _gm.B_scop_unique,
                        "S_scop": S_scop,
                        "bin_idx": _gm.bin_idx,
                        "beta_scop": warm_beta_scop,
                        "beta_scop_prev": None,
                    }
                else:
                    B_scop = _gm.toarray()
                    support = None
                    if _n_scop_groups == 1:
                        support = scop_exact_support.build_exact_scop_support(B_scop)

                    if support is not None:
                        _expose_exact_support_state = True
                        state = {
                            "reparam": reparam,
                            "B_scop": support.B_unique,
                            "S_scop": S_scop,
                            "bin_idx": support.row_to_support,
                            "beta_scop": warm_beta_scop,
                            "beta_scop_prev": None,
                        }
                    else:
                        state = {
                            "reparam": reparam,
                            "B_scop": B_scop,
                            "S_scop": S_scop,
                            "bin_idx": None,
                            "beta_scop": warm_beta_scop,
                            "beta_scop_prev": None,
                        }
                if cached_scop_state is not None:
                    for key in (
                        "H_scop_penalized",
                        "last_step_norm",
                        "last_fisher_fallback",
                        "penalty_rank",
                        "penalty_log_det_omega_plus",
                        "penalty_eigvals_omega",
                    ):
                        if key in cached_scop_state:
                            state[key] = cached_scop_state[key]
                _scop_state[gi] = state
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

        _scop_specs = {
            gi: _SCOPGroupSpec(
                group_index=gi,
                group=groups[gi],
                reparam=st["reparam"],
                B_scop=st["B_scop"],
                S_scop=st["S_scop"],
                bin_idx=st["bin_idx"],
            )
            for gi, st in _scop_state.items()
        }

        # QP/warm initialization is part of the first committed state, not the
        # first proposal. This gives iteration one a coherent latent baseline.
        provisional = _evaluate_irls_state(dm, y, weights, family, link, offset, beta, intercept)
        V_init = np.maximum(family.variance(provisional.mu), _VARIANCE_FLOOR)
        dmu_deta_init = link.deriv_inverse(provisional.eta)
        W_init = weights * dmu_deta_init**2 / V_init
        z_init = provisional.eta + (y - provisional.mu) / dmu_deta_init
        z_off_init = z_init - offset
        for gi, st in _scop_state.items():
            if st["beta_scop"] is None:
                g_i = groups[gi]
                lam_scop = lambda2.get(g_i.name, 0.0) if isinstance(lambda2, dict) else lambda2
                bin_idx = st["bin_idx"]
                if bin_idx is not None:
                    n_bins = st["B_scop"].shape[0]
                    W_agg = np.bincount(bin_idx, weights=W_init, minlength=n_bins)
                    Wz_agg = np.bincount(
                        bin_idx,
                        weights=W_init * z_off_init,
                        minlength=n_bins,
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        z_bin = np.where(W_agg > 0, Wz_agg / W_agg, 0.0)
                    st["beta_scop"] = st["reparam"].qp_initialize(
                        st["B_scop"],
                        z_bin,
                        lambda_penalty=lam_scop,
                        weights=W_agg,
                    )
                else:
                    st["beta_scop"] = st["reparam"].qp_initialize(
                        st["B_scop"],
                        z_off_init,
                        lambda_penalty=lam_scop,
                        weights=W_init,
                    )
            gamma_eff = st["reparam"].forward(st["beta_scop"])
            st["gamma_eff"] = gamma_eff.copy()
            beta[groups[gi].sl] = gamma_eff

    # QR pre-computation: materialise full design matrix once
    # Constrained QP / SCOP requires Gram path — force it if constraints present
    _use_qr = direct_solve == "qr" and not has_constraints and not _has_scop
    if _use_qr:
        has_disc = any(
            isinstance(gm, DiscretizedSSPGroupMatrix | DiscretizedSplineCategoricalGroupMatrix)
            for gm in gms
        )
        if has_disc:
            logger.warning(
                "direct_solve='qr' with discretized groups materialises the full "
                "(n, p) design matrix, defeating the O(n_bins) discretization "
                "benefit.  Consider direct_solve='auto' for large-n discrete fits."
            )
        _X_full = np.hstack([gm.toarray() for gm in gms])  # (n, p)
        _L_aug = _sqrt_penalty_augmented(S, p)  # (p+1, p+1)

    # tabmat acceleration: build SplitMatrix once for non-discrete paths.
    # R_inv is constant within a single fit_irls_direct call, so the
    # materialized X is valid for all IRLS iterations.
    _tabmat_split = dm.tabmat_split if not _use_qr else None
    _can_reuse_weighted_gram = _has_constant_irls_weights(family, link) and not _has_scop
    _constant_w_gram_cache: tuple[NDArray, NDArray, float] | None = None
    _constant_centered_cache: CenteredSystem | None = None
    _constant_centered_z: NDArray | None = None

    def get_centered_system(W_current: NDArray, z_off_current: NDArray) -> CenteredSystem:
        nonlocal _constant_centered_cache, _constant_centered_z
        if (
            _can_reuse_weighted_gram
            and _constant_centered_cache is not None
            and _constant_centered_z is not None
        ):
            if np.array_equal(z_off_current, _constant_centered_z):
                return _constant_centered_cache
            _constant_centered_cache = refresh_centered_rhs(
                system=_constant_centered_cache,
                dm=dm,
                W=W_current,
                z_off=z_off_current,
            )
            _constant_centered_z = z_off_current.copy()
            return _constant_centered_cache
        system = build_centered_system(
            dm=dm,
            W=W_current,
            z_off=z_off_current,
            penalty=S,
        )
        if _can_reuse_weighted_gram:
            _constant_centered_cache = system
            _constant_centered_z = z_off_current.copy()
        return system

    t_start = time.perf_counter()
    converged = False
    XtWX_beta = np.eye(p)  # will be overwritten

    # Phase timing accumulators
    _t_working = 0.0
    _t_gram = 0.0
    _t_solve = 0.0
    _t_deviance = 0.0
    _t_eta = 0.0
    _t_deviance_eval = 0.0
    _last_working_centered: CenteredSystem | None = None

    # Freeze the fit-entry state so iteration-one trial safety has a baseline.
    committed = _evaluate_irls_state(
        dm,
        y,
        weights,
        family,
        link,
        offset,
        beta,
        intercept,
        deviance=_deviance_init,
    )
    if _has_scop:
        scop_committed = _SCOPTrialState(
            irls=committed,
            groups=tuple(
                _SCOPGroupState(
                    group_index=gi,
                    beta_eff=_immutable_array(st["beta_scop"]),
                    gamma_eff=_immutable_array(st["gamma_eff"]),
                    H_scop_penalized=(
                        None
                        if st.get("H_scop_penalized") is None
                        else _immutable_array(st["H_scop_penalized"])
                    ),
                    last_step_norm=float(st.get("last_step_norm", 0.0)),
                    last_fisher_fallback=bool(st.get("last_fisher_fallback", False)),
                )
                for gi, st in sorted(_scop_state.items())
            ),
        )
    dev_prev = committed.deviance
    eta_unclipped = committed.eta_unclipped
    eta = committed.eta
    mu = committed.mu
    iteration_log: list[IterationDiagnostics] = [] if record_diagnostics else []
    base_debug_context = dict(debug_context or {})
    # Level 2 fixes the row schema for the whole fit, so snapshot it at fit entry.
    record_debug_rows = (
        debug_recorder is not None and getattr(debug_recorder, "enabled_level", 0) >= 2
    )
    capture_extrema = record_diagnostics or record_debug_rows

    max_halving = 5  # max step-halving attempts per iteration
    _consecutive_svd = 0  # for auto-mode warning

    for it in range(max_iter):
        beta_prev = committed.beta
        intercept_prev = committed.intercept
        beta = committed.beta.copy()
        intercept = committed.intercept
        committed_active_set = None if prev_active_set is None else list(prev_active_set)
        rank_truncated: bool | None = None

        # Working quantities from current eta/mu (already computed)
        _t0 = time.perf_counter()
        V = family.variance(mu)
        V = np.maximum(V, _VARIANCE_FLOOR)
        dmu_deta = link.deriv_inverse(eta)
        W = weights * dmu_deta**2 / V
        z = eta + (y - mu) / dmu_deta
        working_eta_unclipped = eta_unclipped
        working_eta = eta
        working_mu = mu
        positive_w_min, positive_w_max, w_ratio = _positive_working_weight_stats(W)
        _t_working += time.perf_counter() - _t0

        if w_ratio > 1e12:
            logger.debug(
                "IRLS direct iter=%d: extreme W ratio %.1e (positive W range [%.2e, %.2e])",
                it + 1,
                w_ratio,
                positive_w_min,
                positive_w_max,
            )

        if _use_qr:
            # Profile the intercept, then apply the shared factor-space rule.
            _t0 = time.perf_counter()
            sqrtW = np.sqrt(W)
            z_off = z - offset
            centered = get_centered_system(W, z_off)
            _last_working_centered = centered
            A_data = sqrtW[:, None] * (_X_full - centered.mean_x)
            A = np.vstack([A_data, _L_aug[1:, 1:]])
            rhs_qr = np.concatenate([sqrtW * (z_off - centered.mean_z), np.zeros(p)])
            iteration_rank = decompose_factor(A)
            beta = iteration_rank.solve(A.T @ rhs_qr)
            intercept = centered.mean_z - float(centered.mean_x @ beta)
            _cond_est = iteration_rank.pre_truncation_condition
            _used_svd = iteration_rank.used_svd_fallback
            rank_truncated = iteration_rank.rank_truncated
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
                    gamma_eff = st["reparam"].forward(st["beta_scop"])
                    _eta_g = st["B_scop"] @ gamma_eff
                    if st["bin_idx"] is not None:
                        _eta_g = _eta_g[st["bin_idx"]]
                    eta_scop += _eta_g

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
                    profile=profile,
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

                # Step 6: Apply SCOP Newton step
                if _scop_joint:
                    # Joint Newton step for all SCOP groups simultaneously
                    _z_scop = z_off - eta_unconstrained
                    scop_results = scop_joint_newton_step(
                        _scop_state,
                        W,
                        _z_scop,
                        lambda2,
                        groups,
                        max_halving=10,
                        debug_recorder=debug_recorder,
                        debug_context={
                            **base_debug_context,
                            "pirls_iteration": it + 1,
                        },
                    )
                else:
                    # Sequential (existing code) — for parity comparison
                    scop_results = {}
                    for gi, st in _scop_state.items():
                        z_scop = z_off - eta_unconstrained
                        # Remove contributions from other SCOP groups
                        for gi2, st2 in _scop_state.items():
                            if gi2 != gi:
                                gamma2 = st2["reparam"].forward(st2["beta_scop"])
                                eta2 = st2["B_scop"] @ gamma2
                                if st2["bin_idx"] is not None:
                                    eta2 = eta2[st2["bin_idx"]]
                                z_scop = z_scop - eta2

                        g_i = groups[gi]
                        _lam_scop = (
                            lambda2.get(g_i.name, 0.0) if isinstance(lambda2, dict) else lambda2
                        )
                        result = scop_newton_step(
                            B_scop=st["B_scop"],
                            W=W,
                            z=z_scop,
                            beta_scop=st["beta_scop"],
                            reparam=st["reparam"],
                            S_scop=st["S_scop"],
                            lambda2=_lam_scop,
                            bin_idx=st["bin_idx"],
                            debug_recorder=debug_recorder,
                            debug_context={
                                **base_debug_context,
                                "pirls_iteration": it + 1,
                                "group_name": g_i.name,
                            },
                        )
                        scop_results[gi] = result

                # Step 7: Write gamma_eff (mapped coefficients) into full beta
                for gi, st in _scop_state.items():
                    g = groups[gi]
                    gamma_eff = st["reparam"].forward(scop_results[gi].beta_new)
                    beta[g.sl] = gamma_eff

            elif not has_constraints:
                centered = get_centered_system(W, z_off)
                _last_working_centered = centered
                _t_gram += time.perf_counter() - _t0
                _t0 = time.perf_counter()
                iteration_rank = decompose_gram(centered.hessian)
                if needs_factor_certification(iteration_rank):
                    certified = decompose_factor(
                        grouped_augmented_factor(
                            dm,
                            W,
                            centered.penalty,
                            center=centered.mean_x,
                        )
                    )
                    if certified.rank != iteration_rank.rank:
                        iteration_rank = certified
                beta = iteration_rank.solve(centered.rhs)
                intercept = centered.mean_z - float(centered.mean_x @ beta)
                _cond_est = iteration_rank.pre_truncation_condition
                _used_svd = iteration_rank.used_svd_fallback
                rank_truncated = iteration_rank.rank_truncated
                _t_solve += time.perf_counter() - _t0
            else:
                Wz = W * z_off
                sum_W = float(np.sum(W))

                if _can_reuse_weighted_gram and _constant_w_gram_cache is not None:
                    XtWX, XtW1, sum_W = _constant_w_gram_cache
                    XtWz = dm.rmatvec(Wz)
                else:
                    # Combined gram + rmatvec: shares O(n) bincount for discretized groups
                    XtWX, XtW1, XtWz = _block_xtwx_rhs(
                        gms,
                        groups,
                        W,
                        Wz,
                        tabmat_split=_tabmat_split,
                        profile=profile,
                    )
                    if _can_reuse_weighted_gram:
                        _constant_w_gram_cache = (XtWX, XtW1, sum_W)

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

                # Solve the constrained system in its existing coordinate space.
                _t0 = time.perf_counter()
                # Profile out intercept (unconstrained):
                # intercept = (rhs[0] - XtW1 @ beta) / sum_W
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

        if _has_scop:
            _t0 = time.perf_counter()
            proposal_irls = _evaluate_irls_state(
                dm, y, weights, family, link, offset, beta, intercept
            )
            proposal_scop = _SCOPTrialState(
                irls=proposal_irls,
                groups=tuple(
                    _SCOPGroupState(
                        group_index=gi,
                        beta_eff=_immutable_array(scop_results[gi].beta_new),
                        gamma_eff=_immutable_array(
                            _scop_specs[gi].reparam.forward(scop_results[gi].beta_new)
                        ),
                        H_scop_penalized=(
                            None
                            if scop_results[gi].H_penalized is None
                            else _immutable_array(scop_results[gi].H_penalized)
                        ),
                        last_step_norm=float(scop_results[gi].step_norm),
                        last_fisher_fallback=bool(scop_results[gi].used_fisher_fallback),
                    )
                    for gi in sorted(_scop_specs)
                ),
            )
            scop_trial_cache: dict[float, _SCOPTrialState] = {1.0: proposal_scop}

            def evaluate_scop_trial(alpha: float) -> _IRLSState:
                candidate = _evaluate_scop_trial(
                    committed=scop_committed,
                    proposed=proposal_scop,
                    alpha=alpha,
                    specs=_scop_specs,
                    dm=dm,
                    y=y,
                    weights=weights,
                    family=family,
                    link=link,
                    offset=offset,
                )
                scop_trial_cache[alpha] = candidate
                return candidate.irls

            decision = _select_irls_trial(
                committed=scop_committed.irls,
                proposal=proposal_scop.irls,
                evaluate_state=evaluate_scop_trial,
                max_halving=max_halving,
            )
            retained_scop = (
                scop_committed if decision.step_rejected else scop_trial_cache[decision.alpha]
            )
            retained = retained_scop.irls
            evaluation_elapsed = time.perf_counter() - _t0
            _t_deviance += evaluation_elapsed
            _t_deviance_eval += evaluation_elapsed
            beta = retained.beta.copy()
            intercept = retained.intercept
            eta_unclipped = retained.eta_unclipped
            eta = retained.eta
            mu = retained.mu
            dev = retained.deviance
            n_halvings = decision.step_halvings
            step_rejected = decision.step_rejected
            committed_groups = {group.group_index: group for group in scop_committed.groups}
            for group_state in retained_scop.groups:
                st = _scop_state[group_state.group_index]
                st["beta_scop_prev"] = committed_groups[group_state.group_index].beta_eff.copy()
                st["beta_scop"] = group_state.beta_eff.copy()
                st["gamma_eff"] = group_state.gamma_eff.copy()
                st["H_scop_penalized"] = (
                    None
                    if group_state.H_scop_penalized is None
                    else group_state.H_scop_penalized.copy()
                )
                st["last_step_norm"] = group_state.last_step_norm
                st["last_fisher_fallback"] = group_state.last_fisher_fallback
            if n_halvings:
                logger.info(
                    "  irls_direct SCOP iter=%d: accepted latent step fraction %.5g after "
                    "%d halvings, dev=%.2e",
                    it + 1,
                    decision.alpha,
                    n_halvings,
                    dev,
                )
        else:
            _t0 = time.perf_counter()
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
            evaluation_elapsed = time.perf_counter() - _t0
            _t_deviance += evaluation_elapsed
            _t_deviance_eval += evaluation_elapsed
            beta = retained.beta.copy()
            intercept = retained.intercept
            eta_unclipped = retained.eta_unclipped
            eta = retained.eta
            mu = retained.mu
            dev = retained.deviance
            n_halvings = decision.step_halvings
            step_rejected = decision.step_rejected
            if step_rejected:
                prev_active_set = committed_active_set
            elif n_halvings:
                logger.info(
                    "  irls_direct iter=%d: accepted step fraction %.5g after %d halvings, "
                    "dev=%.2e",
                    it + 1,
                    decision.alpha,
                    n_halvings,
                    dev,
                )

        working_eta_clipped = False
        eta_clipped = False
        if capture_extrema:
            working_eta_clipped = bool(
                float(np.min(working_eta_unclipped)) < float(np.min(working_eta))
                or float(np.max(working_eta_unclipped)) > float(np.max(working_eta))
            )
            eta_clipped = bool(
                float(np.min(eta_unclipped)) < float(np.min(eta))
                or float(np.max(eta_unclipped)) > float(np.max(eta))
            )

        # Record per-iteration diagnostics
        if record_diagnostics:
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
                    raw_w_min=float(W.min()),
                    raw_w_max=float(W.max()),
                    raw_w_ratio=w_ratio,
                    eta_min_unclipped=float(np.min(eta_unclipped)),
                    eta_max_unclipped=float(np.max(eta_unclipped)),
                    eta_clipped=eta_clipped,
                    working_mu_min=float(working_mu.min()),
                    working_mu_max=float(working_mu.max()),
                    working_eta_min=float(working_eta.min()),
                    working_eta_max=float(working_eta.max()),
                    working_eta_min_unclipped=float(np.min(working_eta_unclipped)),
                    working_eta_max_unclipped=float(np.max(working_eta_unclipped)),
                    working_eta_clipped=working_eta_clipped,
                    step_rejected=step_rejected,
                    rank_truncated=rank_truncated,
                )
            )

        logger.info(
            f"  irls_direct iter={it + 1:3d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}"
        )

        if not np.isfinite(dev):
            logger.warning(f"IRLS direct non-finite deviance at iter={it + 1}: dev={dev:.2e}")
            break

        dev_rel_change = None
        coef_change = None
        if convergence == "coefficients":
            coef_change = float(np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta))))
            coef_change = max(
                coef_change,
                abs(intercept - intercept_prev) / max(1.0, abs(intercept)),
            )
            converged_this_iter = coef_change < tol
        else:
            if np.isfinite(dev_prev):
                dev_rel_change = abs(dev - dev_prev) / (abs(dev_prev) + 1.0)
            converged_this_iter = dev_rel_change is not None and dev_rel_change < tol
        if step_rejected:
            converged_this_iter = False

        if record_debug_rows:
            debug_recorder.append_jsonl(
                "pirls",
                {
                    **base_debug_context,
                    "iteration": it + 1,
                    "deviance": float(dev),
                    "deviance_relative_change": (
                        float(dev_rel_change) if dev_rel_change is not None else None
                    ),
                    "coefficient_change": float(coef_change) if coef_change is not None else None,
                    "convergence": convergence,
                    "converged": bool(converged_this_iter),
                    "w_min": float(W.min()),
                    "w_max": float(W.max()),
                    "w_ratio": float(w_ratio),
                    "mu_min": float(mu.min()),
                    "mu_max": float(mu.max()),
                    "eta_min_unclipped": float(np.min(eta_unclipped)),
                    "eta_max_unclipped": float(np.max(eta_unclipped)),
                    "eta_clipped": bool(eta_clipped),
                    "eta_min": float(eta.min()),
                    "eta_max": float(eta.max()),
                    "working_mu_min": float(working_mu.min()),
                    "working_mu_max": float(working_mu.max()),
                    "working_eta_min_unclipped": float(np.min(working_eta_unclipped)),
                    "working_eta_max_unclipped": float(np.max(working_eta_unclipped)),
                    "working_eta_clipped": bool(working_eta_clipped),
                    "working_eta_min": float(working_eta.min()),
                    "working_eta_max": float(working_eta.max()),
                    "step_halvings": int(n_halvings),
                    "step_rejected": bool(step_rejected),
                    "rank_truncated": rank_truncated,
                    "cond_estimate": float(_cond_est),
                    "used_svd_fallback": bool(_used_svd),
                    "has_scop": bool(_has_scop),
                },
            )

        if step_rejected:
            logger.warning(
                "IRLS direct rejected all trial steps at iter=%d; restored committed state",
                it + 1,
            )
            break

        committed = retained
        if _has_scop:
            scop_committed = retained_scop
        if converged_this_iter:
            converged = True
            break
        dev_prev = dev

    t_elapsed = time.perf_counter() - t_start
    logger.info(f"  IRLS direct done: {it + 1} iters, {t_elapsed:.2f}s")

    if _has_scop:
        # Final Gram and SCOP Hessian caches must describe the retained model,
        # not the working state or a discarded full proposal.
        V_final = np.maximum(family.variance(mu), _VARIANCE_FLOOR)
        dmu_deta_final = link.deriv_inverse(eta)
        W = weights * dmu_deta_final**2 / V_final
        z = eta + (y - mu) / dmu_deta_final

        needs_initial_hessian = any(
            state.get("H_scop_penalized") is None for state in _scop_state.values()
        )
        if not step_rejected or needs_initial_hessian:
            eta_unconstrained = np.full(n, intercept)
            for gi in _non_scop_groups_idx:
                g = groups[gi]
                eta_unconstrained += gms[gi].matvec(beta[g.sl])
            z_scop_final = z - offset - eta_unconstrained
            if _scop_joint:
                refresh_results = scop_joint_newton_step(
                    _scop_state,
                    W,
                    z_scop_final,
                    lambda2,
                    groups,
                    max_halving=10,
                )
                for gi, refresh in refresh_results.items():
                    _scop_state[gi]["H_scop_penalized"] = (
                        None if refresh.H_penalized is None else refresh.H_penalized.copy()
                    )
            else:
                for gi, st in _scop_state.items():
                    z_scop_group = z_scop_final.copy()
                    for gi_other, st_other in _scop_state.items():
                        if gi_other == gi:
                            continue
                        eta_other = st_other["B_scop"] @ st_other["gamma_eff"]
                        if st_other["bin_idx"] is not None:
                            eta_other = eta_other[st_other["bin_idx"]]
                        z_scop_group -= eta_other
                    g = groups[gi]
                    lam_scop = lambda2.get(g.name, 0.0) if isinstance(lambda2, dict) else lambda2
                    refresh = scop_newton_step(
                        B_scop=st["B_scop"],
                        W=W,
                        z=z_scop_group,
                        beta_scop=st["beta_scop"],
                        reparam=st["reparam"],
                        S_scop=st["S_scop"],
                        lambda2=lam_scop,
                        bin_idx=st["bin_idx"],
                    )
                    st["H_scop_penalized"] = (
                        None if refresh.H_penalized is None else refresh.H_penalized.copy()
                    )

    # Every public/exported matrix and rank claim is evaluated at the retained
    # model. Private fREML performance iterations deliberately reuse the
    # working system used for their one coefficient update.
    if not _return_working_system:
        V_export = np.maximum(family.variance(mu), _VARIANCE_FLOOR)
        dmu_deta_export = link.deriv_inverse(eta)
        W = weights * dmu_deta_export**2 / V_export
        z = eta + (y - mu) / dmu_deta_export

    # Accumulate phase timing into the profile dict if provided
    if profile is not None:
        profile["irls_working_s"] = profile.get("irls_working_s", 0.0) + _t_working
        profile["irls_gram_s"] = profile.get("irls_gram_s", 0.0) + _t_gram
        profile["irls_solve_s"] = profile.get("irls_solve_s", 0.0) + _t_solve
        profile["irls_deviance_s"] = profile.get("irls_deviance_s", 0.0) + _t_deviance
        profile["irls_eta_s"] = profile.get("irls_eta_s", 0.0) + _t_eta
        profile["irls_deviance_eval_s"] = (
            profile.get("irls_deviance_eval_s", 0.0) + _t_deviance_eval
        )
        profile["irls_total_s"] = profile.get("irls_total_s", 0.0) + t_elapsed
        profile["irls_calls"] = profile.get("irls_calls", 0) + 1
        profile["irls_iters"] = profile.get("irls_iters", 0) + (it + 1)

    # Preserve the raw coefficient-space payload used by REML separately from
    # the centered inference system.
    if _return_working_system:
        if _last_working_centered is None:
            raise RuntimeError("working centered system was not computed")
        centered_final = _last_working_centered
    else:
        z_off = z - offset
        centered_final = get_centered_system(W, z_off)
    XtWX, XtW1, XtWz, sum_Wz = centered_final.raw_weighted_moments()
    sum_W = centered_final.sum_w

    # Cache final-iteration RHS quantities for the cached-W fREML optimizer.
    # These allow re-solving the augmented system with a new penalty matrix S
    # without any data passes (O(p³) instead of O(n·K²) per group).
    if cache_out is not None:
        cache_out["XtWX"] = XtWX
        cache_out["XtWz"] = XtWz
        cache_out["XtW1"] = XtW1
        cache_out["sum_W"] = sum_W
        cache_out["sum_Wz"] = sum_Wz

    # Compute (X'WX + S)^{-1} directly (NOT the intercept-profiled inverse).
    # Reconstruct it from the certified centered Hessian so PSD round-off
    # corrections for degenerate spline penalties remain in force.
    _t0 = time.perf_counter()
    XtWX_beta = XtWX
    M_beta = centered_final.hessian + centered_final.sum_w * np.outer(
        centered_final.mean_x, centered_final.mean_x
    )
    coefficient_rank = decompose_gram(M_beta)
    if needs_factor_certification(coefficient_rank):
        certified = decompose_factor(grouped_augmented_factor(dm, W, centered_final.penalty))
        if certified.rank != coefficient_rank.rank:
            coefficient_rank = certified
    XtWX_S_inv_beta = coefficient_rank.pseudo_inverse()
    log_det_H = coefficient_rank.log_pdet

    if _compute_fit_statistics:
        if _use_qr:
            sqrtW = np.sqrt(W)
            A_data_final = sqrtW[:, None] * (_X_full - centered_final.mean_x)
            data_rank = decompose_factor(A_data_final) if compute_rank_info else None
            augmented_rank = decompose_factor(np.vstack([A_data_final, _L_aug[1:, 1:]]))
        else:
            data_rank = decompose_gram(centered_final.data_gram) if compute_rank_info else None
            if data_rank is not None and needs_factor_certification(data_rank):
                certified = decompose_factor(
                    grouped_weighted_factor(
                        dm,
                        W,
                        center=centered_final.mean_x,
                    )
                )
                if certified.rank != data_rank.rank:
                    data_rank = certified
            augmented_rank = (
                data_rank
                if data_rank is not None and not np.any(centered_final.penalty)
                else decompose_gram(centered_final.hessian)
            )
            if needs_factor_certification(augmented_rank):
                certified = decompose_factor(
                    grouped_augmented_factor(
                        dm,
                        W,
                        centered_final.penalty,
                        center=centered_final.mean_x,
                    )
                )
                if certified.rank != augmented_rank.rank:
                    augmented_rank = certified
        feature_edf = np.diag(augmented_rank.pseudo_inverse() @ centered_final.data_gram).copy()
        feature_edf[np.abs(feature_edf) < 100.0 * np.finfo(float).eps] = 0.0
        p_eff = 1.0 + float(np.sum(feature_edf))
        if compute_rank_info:
            if data_rank is None:
                raise RuntimeError("data-rank metadata was not computed")
            group_edf = {g.name: float(np.sum(feature_edf[g.sl])) for g in groups}
            selected_columns = np.arange(p, dtype=int)
            selected_columns.setflags(write=False)
            feature_edf.setflags(write=False)
            rank_info = RankInfo(
                policy_version=SHARED_RANK_POLICY.version,
                coordinate_space="solver",
                selected_columns=selected_columns,
                selected_group_names=tuple(g.name for g in groups),
                sum_w=centered_final.sum_w,
                mean_x=centered_final.mean_x,
                intercept_edf=1.0,
                data=data_rank,
                augmented=augmented_rank,
                coefficient=coefficient_rank,
                feature_edf=feature_edf,
                group_edf=group_edf,
                objective_loss=None,
            )
        else:
            rank_info = None
    else:
        p_eff = 0.0
        rank_info = None
    if profile is not None:
        _t_finalize = time.perf_counter() - _t0
        profile["irls_finalize_s"] = profile.get("irls_finalize_s", 0.0) + _t_finalize

    # Pearson-based phi for estimated-scale families (Tweedie, Gamma, NB2).
    # SuperGLM's sample_weight follows the prior-weight convention, so the
    # residual d.f. correction is observation-count based (n - edf), while
    # the weights still scale the Pearson numerator.
    if _compute_fit_statistics:
        V_final = np.maximum(family.variance(mu), _VARIANCE_FLOOR)
        pearson_chi2 = float(np.sum(weights * (y - mu) ** 2 / V_final))
        df_resid = max(float(len(y)) - p_eff, 1)
        phi = pearson_chi2 / df_resid
    else:
        phi = 1.0

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
        rank_info=rank_info,
    )

    # Collect converged SCOP state for EFS outer loop and fit results.
    if _has_scop:
        scop_converged = {}
        for gi, st in _scop_state.items():
            scop_converged[gi] = {
                "beta_eff": st["beta_scop"].copy(),
                "H_scop_penalized": st.get("H_scop_penalized"),
                "S_scop": st["S_scop"],
                "B_scop": st["B_scop"],
                "reparam": st["reparam"],
                "gamma_eff": st.get("gamma_eff"),
                "bin_idx": st.get("bin_idx"),
                "group_sl": groups[gi].sl,
                "group_name": groups[gi].name,
                "last_step_norm": st.get("last_step_norm", 0.0),
                "last_fisher_fallback": st.get("last_fisher_fallback", False),
                "penalty_rank": st.get("penalty_rank"),
                "penalty_log_det_omega_plus": st.get("penalty_log_det_omega_plus"),
                "penalty_eigvals_omega": st.get("penalty_eigvals_omega"),
            }
    else:
        scop_converged = None
    if _expose_exact_support_state:
        result.scop_states = scop_converged

    if return_xtwx:
        if return_scop_state and scop_converged is not None:
            return result, XtWX_S_inv_beta, XtWX_beta, scop_converged
        return result, XtWX_S_inv_beta, XtWX_beta

    if return_scop_state and scop_converged is not None:
        return result, XtWX_S_inv_beta, scop_converged
    return result, XtWX_S_inv_beta
