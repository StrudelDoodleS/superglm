"""Cached-W fREML optimizer (discrete path).

Performance Oriented Iteration (mgcv bam-style): interleaves one
PIRLS step (W update) with one Newton lambda step on the working
model's REML criterion.

References
----------
- Wood (2011) Section 6.2.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._fit_trace import TraceRun
from superglm.distributions import Gamma, Gaussian, clip_mu
from superglm.dm_builder import rebuild_design_matrix_with_lambdas
from superglm.group_matrix import DesignMatrix, DiscretizedTensorGroupMatrix
from superglm.links import stabilize_eta
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.penalty_algebra import (
    build_penalty_context,
    build_penalty_matrix,
    build_tensor_pair_logdet_summaries,
    coerce_reml_penalties,
    compute_penalty_nullity,
    compute_total_penalty_rank,
    evaluate_tensor_pair_logdet_summaries,
)
from superglm.reml.result import REMLResult, _map_beta_between_bases
from superglm.reml.scale import (
    GammaScaleProfileData,
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_gram
from superglm.types import GroupSlice, PenaltyComponent


def _solve_cached_profiled_system(
    centered_XtWX: NDArray,
    S: NDArray,
    centered_XtWz: NDArray,
    mean_x: NDArray,
    sum_W: float,
    mean_z: float,
) -> tuple[NDArray, float, float, int]:
    """Solve one cached trial in the authoritative intercept-profiled geometry.

    The stable centered data Gram and RHS are invariant while working weights
    are cached.  Each lambda candidate therefore needs only one ``p x p``
    decomposition of ``H_c = X_c' W X_c + S``.  The same decomposition solves
    the coefficients, applies the shared retained-rank policy, and supplies
    ``log(sum(W)) + log|H_c|_+`` for Wood's REML/LAML criterion.
    """
    if not np.isfinite(sum_W) or sum_W <= 0.0:
        raise ValueError("cached sum_W must be positive and finite")
    hessian = np.asarray(centered_XtWX + S, dtype=np.float64)
    diagonal = np.diag(hessian)
    beta = None
    log_pdet = None
    slope_rank = None
    if np.all(np.isfinite(hessian)) and np.all(diagonal > 0.0):
        column_scale = np.sqrt(diagonal)
        equilibrated = hessian / np.outer(column_scale, column_scale)
        equilibrated = 0.5 * (equilibrated + equilibrated.T)
        try:
            factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
            matrix_norm = float(np.linalg.norm(equilibrated, ord=1))
            reciprocal_condition, info = scipy.linalg.get_lapack_funcs("pocon", (factor,))(
                factor,
                matrix_norm,
                uplo="L",
            )
            safely_full_rank = (
                info == 0
                and np.isfinite(reciprocal_condition)
                and reciprocal_condition
                > SHARED_RANK_POLICY.certification_band * SHARED_RANK_POLICY.gram_rcond
            )
            if safely_full_rank:
                probe = np.arange(1.0, len(diagonal) + 1.0)
                rhs = np.column_stack((centered_XtWz / column_scale, probe))
                solutions = scipy.linalg.cho_solve(
                    (factor, True),
                    rhs,
                    check_finite=False,
                )
                probe_residual = np.linalg.norm(equilibrated @ solutions[:, 1] - probe) / max(
                    np.linalg.norm(probe),
                    np.finfo(float).tiny,
                )
                if probe_residual <= 1.0e-6:
                    beta = solutions[:, 0] / column_scale
                    log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
                        np.sum(np.log(column_scale))
                    )
                    slope_rank = len(diagonal)
        except (np.linalg.LinAlgError, ValueError):
            pass
    if beta is None or log_pdet is None or slope_rank is None:
        decomposition = decompose_gram(hessian)
        beta = decomposition.solve(centered_XtWz)
        log_pdet = decomposition.log_pdet
        slope_rank = decomposition.rank
    intercept = float(mean_z - mean_x @ beta)
    log_det_H = float(np.log(sum_W) + log_pdet)
    return beta, intercept, log_det_H, 1 + slope_rank


def _shared_tensor_group_names(penalties: list[PenaltyComponent], group_matrices: list) -> set[str]:
    grouped: dict[str, list[PenaltyComponent]] = {}
    for pc in penalties:
        grouped.setdefault(pc.group_name, []).append(pc)

    out: set[str] = set()
    for group_name, pcs in grouped.items():
        if len(pcs) <= 1:
            continue
        gm = group_matrices[pcs[0].group_index]
        if isinstance(gm, DiscretizedTensorGroupMatrix):
            out.add(group_name)
    return out


def _shared_tensor_penalty_pairs(
    penalties: list[PenaltyComponent], group_matrices: list
) -> list[tuple[str, tuple[int, int]]]:
    grouped: dict[str, list[int]] = {}
    for i, pc in enumerate(penalties):
        grouped.setdefault(pc.group_name, []).append(i)

    out: list[tuple[str, tuple[int, int]]] = []
    for group_name, idxs in grouped.items():
        if len(idxs) != 2:
            continue
        gm = group_matrices[penalties[idxs[0]].group_index]
        if isinstance(gm, DiscretizedTensorGroupMatrix):
            out.append((group_name, (idxs[0], idxs[1])))
    return out


def optimize_discrete_reml_cached_w(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_ranks: dict[str, float],
    lambdas: dict[str, float],
    *,
    max_reml_iter: int,
    reml_tol: float,
    verbose: bool,
    penalty_caches: dict | None = None,
    profile: dict | None = None,
    direct_solve: str = "auto",
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
    # Legacy kwargs accepted but ignored (removed in POI rewrite)
    max_analytical_per_w: int = 30,
    select_snap: bool = True,
    reml_penalties: list[PenaltyComponent] | None = None,
    estimated_names: set[str] | None = None,
    debug_recorder=None,
    trace_run: TraceRun | None = None,
) -> REMLResult:
    """POI fREML optimizer for the discrete path.

    Performance Oriented Iteration (mgcv bam-style): interleaves one
    PIRLS step (W update) with one Newton lambda step on the working
    model's REML criterion. Line search re-solves the cached, stably centered
    profiled-intercept system analytically (O(p^3), no data pass) for each
    trial lambda.

    Typically converges in 5-15 total iterations instead of the old
    nested architecture's 200+ analytical iterations.

    Note: this is a faster approximate optimizer.  On models with many
    noise features (p >> n_signal), Newton-POI may converge to a
    slightly different REML stationary point than the old Fellner-Schall
    fixed-point path.  The REML surface is flat in noise-feature
    directions, and Newton settles at a nearby minimum where noise
    lambdas are large but not maximally penalized.  Deviance drift is
    typically <0.1% relative (guarded by test_wide_poisson_poi_quality).
    """
    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    scale_known = getattr(distribution, "scale_known", True)
    likelihood_size: float | None = None
    gamma_scale_data: GammaScaleProfileData | None = None
    if isinstance(distribution, Gaussian):
        likelihood_size = float(np.sum(sample_weight, dtype=np.float64))
    elif isinstance(distribution, Gamma):
        gamma_scale_data = prepare_gamma_reml_scale_data(y, sample_weight)
    group_names = [pc.name for pc in penalties]
    m = len(group_names)
    shared_tensor_pairs = _shared_tensor_penalty_pairs(penalties, dm.group_matrices)
    shared_tensor_groups = _shared_tensor_group_names(penalties, dm.group_matrices)
    _t_reml_start = _time.perf_counter()
    _t_pirls = 0.0
    _t_objective = 0.0
    _t_newton = 0.0
    _t_linesearch = 0.0
    _t_linesearch_solve = 0.0
    _t_linesearch_surrogate = 0.0
    _t_linesearch_full_obj = 0.0
    _t_rebuild_dm = 0.0
    _t_map_beta = 0.0
    _t_penalty_context = 0.0
    _t_tensor_summary = 0.0
    penalty_context_cache: dict = {}
    _t0 = _time.perf_counter()
    tensor_pair_summaries = build_tensor_pair_logdet_summaries(
        dm.group_matrices,
        penalties,
        cache=penalty_context_cache,
    )
    _t_tensor_summary += _time.perf_counter() - _t0
    use_tensor_surrogate_linesearch = scale_known and bool(shared_tensor_groups)
    # estimated_mask[i] = True  => component i is free to be optimized
    #                     False => component i has a fixed lambda (policy)
    if estimated_names is not None:
        estimated_mask = np.array([pc.name in estimated_names for pc in penalties])
    else:
        estimated_mask = np.ones(m, dtype=bool)
    log_lo, log_hi = np.log(1e-6), np.log(1e10)
    p = dm.p

    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    warm_beta: NDArray | None = None
    warm_intercept: float | None = None
    warm_deviance: float | None = None
    max_newton_step = 5.0
    max_halving = 25
    _eps = np.finfo(float).eps
    _tol = max(float(reml_tol), 1e-12)

    best_obj = np.inf
    best_lambdas = lambdas.copy()
    best_pirls = None
    converged = False

    _n_pirls_steps = 0
    _n_newton_steps = 0
    _n_linesearch_evals = 0
    _n_linesearch_surrogate_evals = 0
    _n_linesearch_full_evals = 0
    _outer_step_stats: list[dict[str, float | int | bool | None | dict[str, float]]] = []
    _tensor_post_stall_unlocked = False
    _prev_tensor_v: float | None = None

    # === Bootstrap: one FP step from conservative interaction penalties ===
    # Rich tensor interactions can explode under an almost-unpenalized
    # bootstrap fit. Keep main-effect bootstrap lambdas tiny, but start
    # interaction penalty components from a materially stronger seed.
    boot_lambdas = {pc.name: (1.0 if ":" in pc.group_name else 1e-4) for pc in penalties}
    _t0 = _time.perf_counter()
    dm_boot = rebuild_design_matrix_with_lambdas(
        dm,
        groups,
        boot_lambdas,
        sample_weight,
        boot_lambdas,
    )
    _t_rebuild_dm += _time.perf_counter() - _t0
    _t0 = _time.perf_counter()
    penalties_boot, penalty_caches_boot, penalty_ranks_boot = build_penalty_context(
        dm_boot.group_matrices,
        reml_groups,
        cache=penalty_context_cache,
    )
    _t_penalty_context += _time.perf_counter() - _t0
    S_boot = build_penalty_matrix(
        dm_boot.group_matrices,
        groups,
        boot_lambdas,
        p,
        reml_penalties=penalties_boot,
    )
    _pirls_start = _time.perf_counter()
    cache: dict = {}
    boot_result, boot_inv, boot_xtwx = fit_irls_direct(
        X=dm_boot,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        lambda2=boot_lambdas,
        offset=offset_arr,
        max_iter=max_pirls_iter,
        tol=pirls_tol,
        return_xtwx=True,
        compute_rank_info=False,
        _return_working_system=True,
        _compute_fit_statistics=False,
        profile=profile,
        cache_out=cache,
        direct_solve=direct_solve,
        S_override=S_boot,
        debug_recorder=debug_recorder,
        debug_context={"phase": "bootstrap", "reml_iteration": 0},
        trace_run=trace_run,
        trace_purpose="reml_bootstrap",
    )
    _t_pirls += _time.perf_counter() - _pirls_start
    dm = dm_boot
    penalties = penalties_boot
    penalty_caches = penalty_caches_boot
    penalty_ranks = penalty_ranks_boot
    shared_tensor_pairs = _shared_tensor_penalty_pairs(penalties, dm.group_matrices)
    shared_tensor_groups = _shared_tensor_group_names(penalties, dm.group_matrices)
    _t0 = _time.perf_counter()
    tensor_pair_summaries = build_tensor_pair_logdet_summaries(
        dm.group_matrices,
        penalties,
        cache=penalty_context_cache,
    )
    _t_tensor_summary += _time.perf_counter() - _t0
    _n_pirls_steps += boot_result.n_iter
    warm_beta = boot_result.beta.copy()
    warm_intercept = float(boot_result.intercept)
    warm_deviance = float(boot_result.deviance)

    # Bootstrap FP step for initial rho
    boot_phi = 1.0
    boot_inv_phi = 1.0
    boot_penalty_rank_total = compute_total_penalty_rank(penalties)
    if not scale_known and penalty_caches is not None:
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
        if boot_result.reml_hessian_rank is None:
            raise RuntimeError("discrete REML bootstrap is missing full-H rank metadata")
        M_p = compute_penalty_nullity(
            S_boot,
            hessian_rank=boot_result.reml_hessian_rank,
            penalties=penalties,
            lambdas=boot_lambdas,
        )
        penalized_deviance = float(boot_result.deviance + pq_boot)
        if isinstance(distribution, Gaussian):
            assert likelihood_size is not None
            boot_scale = profile_gaussian_reml_scale(
                penalized_deviance,
                likelihood_size,
                M_p,
            )
            boot_phi = boot_scale.phi
            boot_inv_phi = boot_scale.inverse_phi
        elif isinstance(distribution, Gamma):
            assert gamma_scale_data is not None
            boot_scale = profile_gamma_reml_scale(
                gamma_scale_data,
                penalized_deviance,
                M_p,
            )
            boot_phi = boot_scale.phi
            boot_inv_phi = boot_scale.inverse_phi
        else:
            boot_phi = max(
                penalized_deviance / max(len(y) - M_p, 1.0),
                1e-10,
            )
            boot_inv_phi = 1.0 / max(boot_phi, 1e-10)
    else:
        pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
    bootstrap_log_step_cap = 4.0

    # Store original fixed lambda values for exact restoration after exp->clip
    fixed_lambdas: dict[str, float] = {}
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_lambdas[pc.name] = float(lambdas[pc.name])

    rho = np.zeros(m, dtype=np.float64)
    _bootstrap_component_stats: list[dict[str, float | int | str]] = []
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_val = fixed_lambdas[pc.name]
            rho[i] = np.clip(np.log(max(fixed_val, 1e-6)), log_lo, log_hi)
            continue
        omega_ssp = pc.omega_ssp
        if omega_ssp is None:
            gm = dm.group_matrices[pc.group_index]
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        beta_g = boot_result.beta[pc.group_sl]
        quad = float(beta_g @ omega_ssp @ beta_g)
        H_inv_jj = boot_inv[pc.group_sl, pc.group_sl]
        trace_term = float(np.trace(H_inv_jj @ omega_ssp))
        r_j = pc.rank if pc.rank > 0 else (penalty_ranks[pc.name] if penalty_ranks else 0.0)
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        lam_fp_clipped = float(np.clip(lam_fp, 1e-6, 1e10))
        _bootstrap_component_stats.append(
            {
                "name": pc.name,
                "group_name": pc.group_name,
                "rank": float(r_j),
                "quad": quad,
                "trace_term": trace_term,
                "denom": denom,
                "lam_fp_raw": float(lam_fp),
                "lam_fp_clipped": lam_fp_clipped,
                "beta_norm": float(np.linalg.norm(beta_g)),
                "omega_frob": float(np.linalg.norm(omega_ssp)),
                "block_dim": int(beta_g.shape[0]),
            }
        )
        # Snap degenerate select=True null-space penalties to upper bound.
        pc_i = penalties[i]
        if (
            select_snap
            and pc_i.component_type == "selection"
            and trace_term > 1e-12
            and boot_inv_phi * quad < 0.1 * trace_term
        ):
            lam_fp = np.exp(log_hi)
        lam_prev = max(float(lambdas.get(pc.name, 1e-4)), 1e-6)
        log_prev = np.log(lam_prev)
        log_target = np.log(max(lam_fp, 1e-6))
        if ":" in pc.group_name:
            log_target = np.clip(
                log_target,
                log_prev - bootstrap_log_step_cap,
                log_prev + bootstrap_log_step_cap,
            )
        rho[i] = np.clip(log_target, log_lo, log_hi)

    if verbose:
        boot_lam_str = ", ".join(
            f"{name}={np.exp(rho[i]):.4g}" for i, name in enumerate(group_names)
        )
        print(f"  REML bootstrap: lambdas=[{boot_lam_str}]")

    # === POI loop: one PIRLS step + one Newton lambda step ===
    prev_obj = np.inf
    for poi_iter in range(max_reml_iter):
        rho_clipped = np.clip(rho, log_lo, log_hi)
        cand_lambdas = lambdas.copy()
        for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
            cand_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
        cand_lambdas.update(fixed_lambdas)

        # --- Step 1: One PIRLS step (W update) ---
        # Pre-build S once for this candidate
        S_cand = build_penalty_matrix(
            dm.group_matrices,
            groups,
            cand_lambdas,
            p,
            reml_penalties=penalties,
        )

        _t0 = _time.perf_counter()
        cache = {}
        pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=distribution,
            link=link,
            groups=groups,
            lambda2=cand_lambdas,
            offset=offset_arr,
            beta_init=warm_beta,
            intercept_init=warm_intercept,
            max_iter=1,
            tol=pirls_tol,
            return_xtwx=True,
            compute_rank_info=False,
            _return_working_system=True,
            _compute_fit_statistics=False,
            _deviance_init=warm_deviance,
            profile=profile,
            cache_out=cache,
            direct_solve=direct_solve,
            S_override=S_cand,
            debug_recorder=debug_recorder,
            debug_context={"phase": "candidate", "reml_iteration": poi_iter + 1},
            trace_run=trace_run,
            trace_purpose="reml_candidate",
        )
        _t_pirls += _time.perf_counter() - _t0
        _n_pirls_steps += 1
        warm_beta = pirls_result.beta.copy()
        warm_intercept = float(pirls_result.intercept)
        warm_deviance = float(pirls_result.deviance)

        c_centered_XtWX = cache["centered_XtWX"]
        c_centered_XtWz = cache["centered_rhs"]
        c_mean_x = cache["mean_x"]
        c_mean_z = cache["mean_z"]
        c_sum_W = cache["sum_W"]

        # Evaluate REML objective
        _t0 = _time.perf_counter()
        cand_tensor_pair_evals = evaluate_tensor_pair_logdet_summaries(
            tensor_pair_summaries, cand_lambdas
        )
        objective_evaluation = reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            pirls_result,
            cand_lambdas,
            sample_weight,
            offset_arr,
            XtWX=XtWX,
            penalty_caches=penalty_caches,
            log_det_H=pirls_result.log_det_H,
            S_override=S_cand,
            reml_penalties=penalties,
            tensor_pair_evaluations=cand_tensor_pair_evals,
            likelihood_size=likelihood_size,
            gamma_scale_data=gamma_scale_data,
            return_evaluation=True,
        )

        phi_hat = 1.0
        inverse_phi = 1.0
        inverse_phi_derivative = None
        penalty_nullity: float | None = None
        if isinstance(objective_evaluation, REMLObjectiveEvaluation):
            obj = objective_evaluation.value
            penalty_nullity = objective_evaluation.penalty_nullity
            profiled_scale = objective_evaluation.profiled_scale
        else:
            obj = float(objective_evaluation)
            profiled_scale = None
        if not scale_known:
            if profiled_scale is not None:
                phi_hat = profiled_scale.phi
                inverse_phi = profiled_scale.inverse_phi
                inverse_phi_derivative = profiled_scale.d_inverse_phi_d_penalized_deviance
            else:
                if penalty_nullity is None:
                    if pirls_result.reml_hessian_rank is None:
                        raise RuntimeError(
                            "discrete REML iteration is missing full-H rank metadata"
                        )
                    penalty_nullity = compute_penalty_nullity(
                        S_cand,
                        hessian_rank=pirls_result.reml_hessian_rank,
                        penalties=penalties,
                        lambdas=cand_lambdas,
                    )
                if isinstance(objective_evaluation, REMLObjectiveEvaluation):
                    penalized_deviance = objective_evaluation.penalized_deviance
                else:
                    pq = float(pirls_result.beta @ S_cand @ pirls_result.beta)
                    penalized_deviance = float(pirls_result.deviance + pq)
                phi_hat = max(
                    penalized_deviance / max(len(y) - penalty_nullity, 1.0),
                    1e-10,
                )
                inverse_phi = 1.0 / max(phi_hat, 1e-10)
        _t_objective += _time.perf_counter() - _t0
        if trace_run is not None and trace_run.enabled:
            if pirls_result.state_id is None:  # pragma: no cover - trace contract
                raise RuntimeError("traced discrete REML candidate is missing its state ID")
            trace_run.emit_lazy(
                "evaluation",
                lambda: {
                    "state_id": pirls_result.state_id,
                    "evaluation_id": pirls_result.evaluation_id,
                    "solver": "discrete_reml",
                    "phase": "candidate",
                    "outer_iteration": poi_iter + 1,
                    "objective": float(obj),
                    "lambdas": cand_lambdas,
                    "dispersion": float(phi_hat),
                },
                channel="reml",
                purpose="reml_candidate",
                authoritative=False,
            )

        if obj < best_obj:
            best_obj = obj
            best_lambdas = cand_lambdas.copy()
            best_pirls = pirls_result
        lambda_history.append(cand_lambdas.copy())

        # --- Step 2: Newton step on lambda ---
        _t0 = _time.perf_counter()
        grad = reml_direct_gradient(
            dm.group_matrices,
            pirls_result,
            XtWX_S_inv,
            cand_lambdas,
            reml_penalties=penalties,
            phi_hat=phi_hat,
            inverse_phi=inverse_phi,
            tensor_pair_evaluations=cand_tensor_pair_evals,
        )
        hess = reml_direct_hessian(
            dm.group_matrices,
            distribution,
            XtWX_S_inv,
            cand_lambdas,
            gradient=grad,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
            inverse_phi=inverse_phi,
            d_inverse_phi_d_penalized_deviance=inverse_phi_derivative,
            penalty_nullity=penalty_nullity if not scale_known else None,
            reml_penalties=penalties,
            tensor_pair_evaluations=cand_tensor_pair_evals,
        )

        # Active-set: freeze components with negligible gradient and Hessian
        # Wood (2011) Section 6.2: score_scale = 1 + |V_r|
        score_scale_d = max(1.0 + abs(obj), 1.0)
        freeze_tol_d = 0.1 * _tol

        proj_grad_d = grad.copy()
        for i in range(m):
            if not estimated_mask[i]:
                # Fixed lambda — always zero out gradient contribution
                proj_grad_d[i] = 0.0
            elif rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                proj_grad_d[i] = 0.0
            elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                proj_grad_d[i] = 0.0

        frozen_d = np.zeros(m, dtype=bool)
        for i in range(m):
            if not estimated_mask[i]:
                # Fixed lambda — always freeze
                frozen_d[i] = True
            elif (
                abs(proj_grad_d[i]) < freeze_tol_d * score_scale_d
                and abs(hess[i, i]) < freeze_tol_d * score_scale_d
            ):
                frozen_d[i] = True
        active_idx_d = np.where(~frozen_d)[0]

        # Modified Newton: eigendecompose, flip negatives, floor small eigenvalues
        if active_idx_d.size == 0:
            delta = np.zeros(m)
        else:
            if active_idx_d.size < m:
                hess_sub_d = hess[np.ix_(active_idx_d, active_idx_d)]
                grad_sub_d = grad[active_idx_d]
            else:
                hess_sub_d = hess
                grad_sub_d = grad

            eigvals_h, eigvecs_h = np.linalg.eigh(hess_sub_d)
            max_eig_d = max(abs(eigvals_h).max(), 1e-12)
            eig_floor_d = max_eig_d * _eps**0.7
            eigvals_pd = np.maximum(np.abs(eigvals_h), eig_floor_d)
            delta_sub_d = -(eigvecs_h * (1.0 / eigvals_pd)) @ (eigvecs_h.T @ grad_sub_d)
            delta = np.zeros(m)
            delta[active_idx_d] = delta_sub_d

        tensor_step_diag = None
        if use_tensor_surrogate_linesearch:
            base_cap = 1.0 if not _tensor_post_stall_unlocked else 2.5
            delta = np.clip(delta, -base_cap, base_cap)
            for group_name, (i, j) in shared_tensor_pairs:
                if not estimated_mask[i] or not estimated_mask[j]:
                    continue
                grad_pair = grad[[i, j]]
                hess_pair = hess[np.ix_([i, j], [i, j])]
                J = np.array([[1.0, 1.0], [1.0, -1.0]])
                grad_uv = J.T @ grad_pair
                hess_uv = J.T @ hess_pair @ J
                eigvals_uv, eigvecs_uv = np.linalg.eigh(hess_uv)
                max_eig_uv = max(abs(eigvals_uv).max(), 1e-12)
                eig_floor_uv = max_eig_uv * _eps**0.7
                eigvals_uv_pd = np.maximum(np.abs(eigvals_uv), eig_floor_uv)
                delta_uv = -(eigvecs_uv * (1.0 / eigvals_uv_pd)) @ (eigvecs_uv.T @ grad_uv)
                raw_u = float(delta_uv[0])
                raw_v = float(delta_uv[1])
                cap_u = 2.5 if not _tensor_post_stall_unlocked else 5.0
                cap_v = 0.25 if not _tensor_post_stall_unlocked else 0.35
                used_u = float(np.clip(raw_u, -cap_u, cap_u))
                used_v = float(np.clip(raw_v, -cap_v, cap_v))
                delta_pair = J @ np.array([used_u, used_v])
                delta[i] = float(delta_pair[0])
                delta[j] = float(delta_pair[1])
                if tensor_step_diag is None:
                    tensor_step_diag = {
                        "group_name": group_name,
                        "delta_u_raw": raw_u,
                        "delta_u_used": used_u,
                        "delta_v_raw": raw_v,
                        "delta_v_used": used_v,
                    }

        # Step capping. Shared discrete tensor penalties are especially
        # sensitive to oversized log-lambda steps: they trigger many surrogate
        # halvings even after the cheap trial path is in place. Keep their
        # trust region much tighter than the generic path.
        if not use_tensor_surrogate_linesearch:
            local_max_newton_step = max_newton_step
            max_delta = float(np.max(np.abs(delta)))
            max_delta_raw = max_delta
            if max_delta > local_max_newton_step:
                delta *= local_max_newton_step / max_delta
        else:
            max_delta = float(np.max(np.abs(delta)))
            max_delta_raw = max_delta
        max_delta_raw = max_delta
        quad_grad = float(grad @ delta) if use_tensor_surrogate_linesearch else 0.0
        quad_curv = float(delta @ hess @ delta) if use_tensor_surrogate_linesearch else 0.0
        _t_newton += _time.perf_counter() - _t0
        _n_newton_steps += 1

        # --- Step 3: Line search (step halving on working-model REML) ---
        _t0 = _time.perf_counter()
        accepted = False
        step = 1.0
        candidate = None
        halving_count = 0
        local_max_halving = 5 if use_tensor_surrogate_linesearch else max_halving
        if use_tensor_surrogate_linesearch and max_delta_raw < 1e-12:
            local_max_halving = 0
        for _ls in range(local_max_halving):
            rho_trial = np.clip(rho + step * delta, log_lo, log_hi)
            trial_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_trial), strict=False):
                trial_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
            trial_lambdas.update(fixed_lambdas)

            # Build the trial penalty before the cached profiled solve.
            S_trial = build_penalty_matrix(
                dm.group_matrices,
                groups,
                trial_lambdas,
                p,
                reml_penalties=penalties,
            )

            _n_linesearch_evals += 1
            if use_tensor_surrogate_linesearch:
                _tls0 = _time.perf_counter()
                trial_quad_obj = obj + step * quad_grad + 0.5 * (step**2) * quad_curv
                _t_linesearch_surrogate += _time.perf_counter() - _tls0
                _n_linesearch_surrogate_evals += 1
                if trial_quad_obj >= obj:
                    step *= 0.5
                    halving_count += 1
                    continue
                candidate = (
                    rho_trial,
                    trial_lambdas,
                    S_trial,
                )
                break

            # Solve the cached profiled-intercept system analytically
            # (O(p^3), no data pass).
            _tls0 = _time.perf_counter()
            beta_trial, intercept_trial, log_det_H_trial, hessian_rank_trial = (
                _solve_cached_profiled_system(
                    c_centered_XtWX,
                    S_trial,
                    c_centered_XtWz,
                    c_mean_x,
                    c_sum_W,
                    c_mean_z,
                )
            )
            _t_linesearch_solve += _time.perf_counter() - _tls0

            # Evaluate full REML at trial point once the cached surrogate
            # suggests an improving direction (or for all trials on the
            # non-tensor / estimated-scale path).
            eta_trial = stabilize_eta(dm.matvec(beta_trial) + intercept_trial + offset_arr, link)
            mu_trial = clip_mu(link.inverse(eta_trial), distribution)
            dev_trial = float(np.sum(sample_weight * distribution.deviance_unit(y, mu_trial)))
            trial_pirls = PIRLSResult(
                beta=beta_trial,
                intercept=intercept_trial,
                deviance=dev_trial,
                n_iter=0,
                converged=True,
                phi=phi_hat,
                effective_df=0.0,
                log_det_H=log_det_H_trial,
                reml_hessian_rank=hessian_rank_trial,
            )
            trial_tensor_pair_evals = evaluate_tensor_pair_logdet_summaries(
                tensor_pair_summaries, trial_lambdas
            )
            trial_obj = reml_laml_objective(
                dm,
                distribution,
                link,
                groups,
                y,
                trial_pirls,
                trial_lambdas,
                sample_weight,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
                log_det_H=log_det_H_trial,
                S_override=S_trial,
                reml_penalties=penalties,
                tensor_pair_evaluations=trial_tensor_pair_evals,
                likelihood_size=likelihood_size,
                gamma_scale_data=gamma_scale_data,
            )
            _n_linesearch_full_evals += 1

            if trial_obj < obj:
                rho = rho_trial
                warm_beta = beta_trial.copy()
                warm_intercept = intercept_trial
                warm_deviance = dev_trial
                accepted = True
                break

            step *= 0.5
            halving_count += 1

        if use_tensor_surrogate_linesearch and candidate is not None and not accepted:
            rho_trial, trial_lambdas, S_trial = candidate
            _tls0 = _time.perf_counter()
            beta_trial, intercept_trial, log_det_H_trial, hessian_rank_trial = (
                _solve_cached_profiled_system(
                    c_centered_XtWX,
                    S_trial,
                    c_centered_XtWz,
                    c_mean_x,
                    c_sum_W,
                    c_mean_z,
                )
            )
            eta_trial = stabilize_eta(dm.matvec(beta_trial) + intercept_trial + offset_arr, link)
            mu_trial = clip_mu(link.inverse(eta_trial), distribution)
            dev_trial = float(np.sum(sample_weight * distribution.deviance_unit(y, mu_trial)))
            trial_pirls = PIRLSResult(
                beta=beta_trial,
                intercept=intercept_trial,
                deviance=dev_trial,
                n_iter=0,
                converged=True,
                phi=phi_hat,
                effective_df=0.0,
                log_det_H=log_det_H_trial,
                reml_hessian_rank=hessian_rank_trial,
            )
            trial_tensor_pair_evals = evaluate_tensor_pair_logdet_summaries(
                tensor_pair_summaries, trial_lambdas
            )
            trial_obj = reml_laml_objective(
                dm,
                distribution,
                link,
                groups,
                y,
                trial_pirls,
                trial_lambdas,
                sample_weight,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
                log_det_H=log_det_H_trial,
                S_override=S_trial,
                reml_penalties=penalties,
                tensor_pair_evaluations=trial_tensor_pair_evals,
                likelihood_size=likelihood_size,
                gamma_scale_data=gamma_scale_data,
            )
            _t_linesearch_full_obj += _time.perf_counter() - _tls0
            _n_linesearch_full_evals += 1
            if trial_obj < obj:
                rho = rho_trial
                warm_beta = beta_trial.copy()
                warm_intercept = intercept_trial
                warm_deviance = dev_trial
                accepted = True

        _t_linesearch += _time.perf_counter() - _t0
        if use_tensor_surrogate_linesearch and accepted and halving_count == 0:
            _tensor_post_stall_unlocked = True

        if not accepted:
            # Steepest descent fallback: unit-length in infinity norm
            # Use proj_grad_d so that fixed components are not moved.
            grad_max_d = float(np.max(np.abs(proj_grad_d)))
            if grad_max_d > 1e-12:
                rho = np.clip(
                    rho - proj_grad_d / grad_max_d,
                    log_lo,
                    log_hi,
                )
            # else: keep rho unchanged

        # Convergence check -- compound criterion with score_scale
        proj_grad_norm = float(np.max(np.abs(proj_grad_d)))
        if use_tensor_surrogate_linesearch:
            tensor_names = [pc.name for pc in penalties if pc.group_name in shared_tensor_groups]
            tensor_lams = {name: float(cand_lambdas[name]) for name in tensor_names}
            tensor_log_ratio = None
            if len(tensor_names) == 2:
                tensor_log_ratio = float(
                    np.log(max(cand_lambdas[tensor_names[0]], 1e-12))
                    - np.log(max(cand_lambdas[tensor_names[1]], 1e-12))
                )
            _outer_step_stats.append(
                {
                    "iter": poi_iter + 1,
                    "grad_norm": proj_grad_norm,
                    "max_delta_raw": max_delta_raw,
                    "max_delta_used": float(np.max(np.abs(delta))),
                    "accepted_step": step if accepted else 0.0,
                    "halvings": halving_count,
                    "accepted": accepted,
                    "tensor_log_ratio": tensor_log_ratio,
                    "tensor_lambdas": tensor_lams,
                    "tensor_uv": tensor_step_diag,
                    "tensor_v_sign_flip": (
                        None
                        if tensor_log_ratio is None or _prev_tensor_v is None
                        else bool(
                            (_prev_tensor_v > 0 and tensor_log_ratio < 0)
                            or (_prev_tensor_v < 0 and tensor_log_ratio > 0)
                        )
                    ),
                }
            )
            if tensor_log_ratio is not None:
                _prev_tensor_v = tensor_log_ratio

        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            obj_change_d = abs(obj - prev_obj) if poi_iter > 0 else np.inf
            print(
                f"  POI iter {poi_iter + 1}  obj={obj:.4f}  "
                f"|grad|={proj_grad_norm:.6f}  delta_obj={obj_change_d:.6g}  [{lam_str}]"
            )

        obj_change = abs(obj - prev_obj) if poi_iter > 0 else np.inf
        prev_obj = obj
        if poi_iter >= 1:
            grad_converged_d = proj_grad_norm < _tol * score_scale_d
            obj_converged_d = obj_change < _tol * score_scale_d
            if grad_converged_d and obj_converged_d:
                converged = True
                break

        current_lambdas = lambdas.copy()
        for name, val in zip(group_names, np.exp(np.clip(rho, log_lo, log_hi)), strict=False):
            current_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
        current_lambdas.update(fixed_lambdas)

        old_gms = dm.group_matrices
        _t0 = _time.perf_counter()
        dm = rebuild_design_matrix_with_lambdas(
            dm,
            groups,
            current_lambdas,
            sample_weight,
            current_lambdas,
        )
        _t_rebuild_dm += _time.perf_counter() - _t0
        _t0 = _time.perf_counter()
        warm_beta = _map_beta_between_bases(
            pirls_result.beta,
            old_gms,
            dm.group_matrices,
            groups,
        )
        _t_map_beta += _time.perf_counter() - _t0
        warm_intercept = float(pirls_result.intercept)
        warm_deviance = float(pirls_result.deviance)
        _t0 = _time.perf_counter()
        penalties, penalty_caches, penalty_ranks = build_penalty_context(
            dm.group_matrices,
            reml_groups,
            cache=penalty_context_cache,
        )
        _t_penalty_context += _time.perf_counter() - _t0
        shared_tensor_pairs = _shared_tensor_penalty_pairs(penalties, dm.group_matrices)
        shared_tensor_groups = _shared_tensor_group_names(penalties, dm.group_matrices)
        _t0 = _time.perf_counter()
        tensor_pair_summaries = build_tensor_pair_logdet_summaries(
            dm.group_matrices,
            penalties,
            cache=penalty_context_cache,
        )
        _t_tensor_summary += _time.perf_counter() - _t0

    # === Final full IRLS refit at converged lambdas ===
    rho_clipped = np.clip(rho, log_lo, log_hi)
    final_lambdas = lambdas.copy()
    for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
        final_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
    final_lambdas.update(fixed_lambdas)
    old_gms_final = dm.group_matrices
    _t0 = _time.perf_counter()
    dm = rebuild_design_matrix_with_lambdas(
        dm,
        groups,
        final_lambdas,
        sample_weight,
        final_lambdas,
    )
    _t_rebuild_dm += _time.perf_counter() - _t0
    _t0 = _time.perf_counter()
    warm_beta = _map_beta_between_bases(
        warm_beta if warm_beta is not None else pirls_result.beta,
        old_gms_final,
        dm.group_matrices,
        groups,
    )
    _t_map_beta += _time.perf_counter() - _t0
    _t0 = _time.perf_counter()
    penalties, penalty_caches, penalty_ranks = build_penalty_context(
        dm.group_matrices,
        reml_groups,
        cache=penalty_context_cache,
    )
    _t_penalty_context += _time.perf_counter() - _t0
    shared_tensor_pairs = _shared_tensor_penalty_pairs(penalties, dm.group_matrices)
    shared_tensor_groups = _shared_tensor_group_names(penalties, dm.group_matrices)
    _t0 = _time.perf_counter()
    tensor_pair_summaries = build_tensor_pair_logdet_summaries(
        dm.group_matrices,
        penalties,
        cache=penalty_context_cache,
    )
    _t_tensor_summary += _time.perf_counter() - _t0
    S_final = build_penalty_matrix(
        dm.group_matrices, groups, final_lambdas, dm.p, reml_penalties=penalties
    )
    _t0 = _time.perf_counter()
    final_result, final_inv, final_xtwx = fit_irls_direct(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        lambda2=final_lambdas,
        offset=offset_arr,
        beta_init=warm_beta,
        intercept_init=warm_intercept,
        max_iter=max_pirls_iter,
        tol=pirls_tol,
        return_xtwx=True,
        profile=profile,
        direct_solve=direct_solve,
        S_override=S_final,
        debug_recorder=debug_recorder,
        debug_context={"phase": "optimizer_final", "reml_iteration": poi_iter + 1},
        trace_run=trace_run,
        trace_purpose="reml_optimizer_final",
    )
    _t_pirls += _time.perf_counter() - _t0
    _t0 = _time.perf_counter()
    final_tensor_pair_evals = evaluate_tensor_pair_logdet_summaries(
        tensor_pair_summaries, final_lambdas
    )
    final_obj = reml_laml_objective(
        dm,
        distribution,
        link,
        groups,
        y,
        final_result,
        final_lambdas,
        sample_weight,
        offset_arr,
        XtWX=final_xtwx,
        penalty_caches=penalty_caches,
        log_det_H=final_result.log_det_H,
        S_override=S_final,
        reml_penalties=penalties,
        tensor_pair_evaluations=final_tensor_pair_evals,
        likelihood_size=likelihood_size,
        gamma_scale_data=gamma_scale_data,
    )
    _t_objective += _time.perf_counter() - _t0
    # Always use the final refit -- it is the authoritative result from
    # full IRLS convergence at the converged lambdas.  The working-model
    # surrogates from the POI loop (n_iter=0) must not leak out.
    best_obj = final_obj
    best_lambdas = final_lambdas.copy()
    best_pirls = final_result
    lambda_history.append(final_lambdas.copy())

    if profile is not None:
        if _bootstrap_component_stats:
            profile["reml_bootstrap_summary"] = {
                "boot_phi": float(boot_phi),
                "boot_inv_phi": float(boot_inv_phi),
                "boot_deviance": float(boot_result.deviance),
                "boot_penalty_quad": float(pq_boot),
                "boot_penalty_rank_total": float(boot_penalty_rank_total),
                "n_components": len(_bootstrap_component_stats),
                "lam_fp_min": float(
                    min(row["lam_fp_clipped"] for row in _bootstrap_component_stats)
                ),
                "lam_fp_max": float(
                    max(row["lam_fp_clipped"] for row in _bootstrap_component_stats)
                ),
                "n_components_at_lower_bound": int(
                    sum(row["lam_fp_clipped"] <= 1.0000001e-6 for row in _bootstrap_component_stats)
                ),
            }
            profile["reml_bootstrap_components"] = _bootstrap_component_stats
        profile["reml_optimizer_s"] = _time.perf_counter() - _t_reml_start
        profile["reml_pirls_s"] = _t_pirls
        profile["reml_objective_s"] = _t_objective
        profile["reml_gradient_s"] = 0.0
        profile["reml_w_correction_s"] = 0.0
        profile["reml_hessian_newton_s"] = _t_newton
        profile["reml_linesearch_s"] = _t_linesearch
        profile["reml_linesearch_solve_s"] = _t_linesearch_solve
        profile["reml_linesearch_surrogate_s"] = _t_linesearch_surrogate
        profile["reml_linesearch_full_obj_s"] = _t_linesearch_full_obj
        profile["reml_rebuild_dm_s"] = _t_rebuild_dm
        profile["reml_map_beta_s"] = _t_map_beta
        profile["reml_penalty_context_s"] = _t_penalty_context
        profile["reml_tensor_summary_s"] = _t_tensor_summary
        profile["reml_fp_update_s"] = 0.0
        profile["reml_n_linesearch_fits"] = _n_linesearch_evals
        profile["reml_n_linesearch_surrogate_evals"] = _n_linesearch_surrogate_evals
        profile["reml_n_linesearch_full_evals"] = _n_linesearch_full_evals
        profile["reml_n_outer_iter"] = poi_iter + 1
        profile["reml_n_analytical_iters"] = _n_newton_steps
        if _outer_step_stats:
            profile["reml_outer_step_stats"] = _outer_step_stats

    return REMLResult(
        lambdas=best_lambdas,
        pirls_result=best_pirls,
        n_reml_iter=poi_iter + 1,
        converged=converged,
        lambda_history=lambda_history,
        objective=float(best_obj),
        curvature_source="fisher",
    )
