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
from superglm.distributions import Gamma, Gaussian, Tweedie, clip_mu
from superglm.dm_builder import rebuild_design_matrix_with_lambdas
from superglm.group_matrix import DesignMatrix, DiscretizedTensorGroupMatrix
from superglm.links import stabilize_eta
from superglm.reml.convergence import (
    FLAT_DIRECTION_FREEZE_FLOOR,
    classify_dead_feasible_exit,
    direction_penalty_ranks,
    evaluate_reml_candidate,
    freeze_flat_directions,
    mask_frozen_stop_gradient,
    project_reml_gradient,
    trial_counts_as_precision_evidence,
)
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import (
    REMLObjectiveEvaluation,
    reml_laml_objective,
)
from superglm.reml.penalty_algebra import (
    build_penalty_context,
    build_penalty_matrix,
    build_tensor_pair_logdet_summaries,
    coerce_reml_penalties,
    compute_penalty_nullity,
    compute_total_penalty_rank,
    evaluate_tensor_pair_logdet_summaries,
    penalty_component_quadratic,
    total_penalty_quadratic,
)
from superglm.reml.result import REMLResult, _map_beta_between_bases
from superglm.reml.scale import (
    prepare_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
    profile_tweedie_reml_scale,
)
from superglm.solvers.hessian_factor import as_hessian_factor
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_gram
from superglm.solvers.structured import (
    BlockStructuredSystem,
    ScalarStructuredSystem,
    SumToZeroBlockStructuredSystem,
    record_auto_backend_decision,
    resolve_structured_backend,
    solve_cached_structured,
)
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


def _tensor_trust_ratios(
    delta: NDArray,
    names: list[str],
    shared_tensor_pairs: list[tuple[str, tuple[int, int]]],
    frozen: NDArray,
    base_cap: float,
    cap_v: float,
) -> list[tuple[str, float]]:
    """Constraint ratios of ``delta`` against the shared-tensor trust region.

    The region is the box ``|delta_k| <= base_cap`` on every coordinate,
    intersected for every shared tensor pair whose two margins are both
    active with ``|(d_i - d_j) / 2| <= cap_v``. The coordinate box already
    bounds ``|(d_i + d_j) / 2|`` by ``base_cap``. A ratio at or below 1.0
    is feasible.
    """
    ratios = [(f"base_cap:{names[k]}", abs(float(delta[k])) / base_cap) for k in range(delta.size)]
    for group_name, (i, j) in shared_tensor_pairs:
        if frozen[i] or frozen[j]:
            continue
        v = 0.5 * (float(delta[i]) - float(delta[j]))
        ratios.append((f"{group_name}:v", abs(v) / cap_v))
    return ratios


def _damped_tensor_newton_step(
    eigvecs: NDArray,
    eigvals_pd: NDArray,
    grad_sub: NDArray,
    active_idx: NDArray,
    m: int,
    names: list[str],
    shared_tensor_pairs: list[tuple[str, tuple[int, int]]],
    frozen: NDArray,
    base_cap: float,
    cap_v: float,
) -> tuple[NDArray, float, list[str]]:
    """Damped modified-Newton step inside the shared-tensor trust region.

    With positive eigenvalues, ``delta(mu) = -(H_pd + mu I)^-1 g`` has
    ``g . delta < 0`` for every nonzero gradient and ``mu >= 0``. Unlike
    coordinate clipping, damping preserves that descent property.

    Try the undamped step first, then bracket a feasible damping value
    using ``|delta|_2 <= |g|_2 / mu`` and refine its infeasible/feasible
    bracket on ``log mu``. Individual constraint values need not be
    monotone in mu, so this finds neither the smallest feasible damping
    globally nor the exact box-constrained minimizer. Each trial costs
    ``O(q^2)`` on the active subspace. Return the step, damping value and
    binding constraints. See the September 19 tensor-step research note.
    """
    vt_g = eigvecs.T @ grad_sub

    def step_at(mu: float) -> NDArray:
        delta = np.zeros(m)
        delta[active_idx] = -(eigvecs * (1.0 / (eigvals_pd + mu))) @ vt_g
        return delta

    def worst(delta: NDArray) -> float:
        ratios = _tensor_trust_ratios(delta, names, shared_tensor_pairs, frozen, base_cap, cap_v)
        return max((r for _, r in ratios), default=0.0)

    def binding(delta: NDArray) -> list[str]:
        ratios = _tensor_trust_ratios(delta, names, shared_tensor_pairs, frozen, base_cap, cap_v)
        return [name for name, r in ratios if r >= 1.0 - 1e-6]

    delta = step_at(0.0)
    if worst(delta) <= 1.0:
        return delta, 0.0, binding(delta)

    pair_active = any(not (frozen[i] or frozen[j]) for _, (i, j) in shared_tensor_pairs)
    radius = min(base_cap, cap_v) if pair_active else base_cap
    mu_hi = max(float(np.linalg.norm(grad_sub)) / radius, np.finfo(float).tiny)
    for _ in range(64):  # round-off guard on the analytic bound
        if worst(step_at(mu_hi)) <= 1.0:
            break
        mu_hi *= 2.0
    else:  # pragma: no cover - delta(mu) -> 0 as mu -> inf
        raise RuntimeError(
            "Discrete tensor trust region could not be satisfied by damping: "
            f"mu={mu_hi:.6g}, violation ratio={worst(step_at(mu_hi)):.6g}."
        )
    mu_lo = 0.5 * mu_hi
    for _ in range(64):
        if worst(step_at(mu_lo)) > 1.0:
            break
        mu_hi = mu_lo
        mu_lo *= 0.5
    else:
        delta = step_at(mu_hi)
        return delta, mu_hi, binding(delta)
    for _ in range(64):
        if mu_hi <= mu_lo * (1.0 + 1e-9):
            break
        mid = float(np.sqrt(mu_lo * mu_hi))
        if worst(step_at(mid)) <= 1.0:
            mu_hi = mid
        else:
            mu_lo = mid
    delta = step_at(mu_hi)
    return delta, mu_hi, binding(delta)


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
    weight_semantics: str,
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

    # The declared contract's likelihood size, computed once. Every dispersion
    # denominator and the REML scale term's `0.5 * (n - M_p) * log(D)` must use
    # THIS, not the physical row count: `sum(w)` under `"frequency"`, the
    # positive-row count under `"prior"`. The objective, its gradient and its
    # Hessian all read it, so a row count in any one of them makes the Newton
    # step inconsistent with the surface it is stepping on.
    _contract_size_cache: list[float] = []

    def _contract_size() -> float:
        if not _contract_size_cache:
            from superglm.solvers.dispersion import dispersion_likelihood_size

            _contract_size_cache.append(
                dispersion_likelihood_size(sample_weight, weight_semantics=weight_semantics)
            )
        return _contract_size_cache[0]

    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    scale_known = getattr(distribution, "scale_known", True)
    (
        likelihood_size,
        saturated_log_weight,
        gamma_scale_data,
        tweedie_scale_data,
    ) = prepare_reml_scale_data(
        distribution,
        y,
        sample_weight,
        weight_semantics=weight_semantics,
    )
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
    _t_structured_cache_solve = 0.0
    _t_block_structured_cache_solve = 0.0
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
    structured_decision = resolve_structured_backend(
        list(dm.group_matrices),
        groups,
        direct_solve=direct_solve,
        coefficient_width=p,
        row_weights=sample_weight,
        lambda2=lambdas,
    )
    use_structured = structured_decision.use_structured
    record_auto_backend_decision(profile, direct_solve, structured_decision)

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
    termination_reason = "max_reml_iter"
    all_lambdas_fixed = not bool(np.any(estimated_mask))

    _n_pirls_steps = 0
    _n_newton_steps = 0
    _n_linesearch_evals = 0
    _n_structured_cache_solves = 0
    _n_block_structured_cache_solves = 0
    _n_linesearch_surrogate_evals = 0
    _n_linesearch_full_evals = 0
    _n_dead_line_searches = 0
    _outer_step_stats: list[dict[str, Any]] = []
    _tensor_post_stall_unlocked = False
    _prev_tensor_v: float | None = None
    structured_runtime_fallback_reason: str | None = None

    def latch_runtime_backend(
        pirls_result: PIRLSResult,
        lambda_values: dict[str, float],
        penalty: NDArray | None,
        *,
        design: DesignMatrix,
        penalty_components: list[PenaltyComponent],
    ) -> NDArray | None:
        """Pin later REML work to Gram after an automatic structured retry."""
        nonlocal direct_solve, structured_runtime_fallback_reason, use_structured
        if not use_structured or pirls_result.direct_backend == "structured":
            return penalty
        use_structured = False
        direct_solve = "gram"
        structured_runtime_fallback_reason = pirls_result.direct_fallback_reason
        if penalty is not None:
            return penalty
        return build_penalty_matrix(
            list(design.group_matrices),
            groups,
            lambda_values,
            design.p,
            reml_penalties=penalty_components,
        )

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
    S_boot = (
        None
        if use_structured
        else build_penalty_matrix(
            dm_boot.group_matrices,
            groups,
            boot_lambdas,
            p,
            reml_penalties=penalties_boot,
        )
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
        reml_penalties=penalties_boot,
        debug_recorder=debug_recorder,
        debug_context={"phase": "bootstrap", "reml_iteration": 0},
        trace_run=trace_run,
        trace_purpose="reml_bootstrap",
        weight_semantics=weight_semantics,
    )
    _t_pirls += _time.perf_counter() - _pirls_start
    S_boot = latch_runtime_backend(
        boot_result,
        boot_lambdas,
        S_boot,
        design=dm_boot,
        penalty_components=penalties_boot,
    )
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
        pq_boot = total_penalty_quadratic(
            boot_result.beta,
            boot_lambdas,
            penalties,
            list(dm.group_matrices),
        )
        if boot_result.reml_hessian_rank is None:
            raise RuntimeError("discrete REML bootstrap is missing full-H rank metadata")
        M_p = compute_penalty_nullity(
            S_boot,
            hessian_rank=boot_result.reml_hessian_rank,
            penalties=penalties,
            lambdas=boot_lambdas,
            coefficient_width=p,
        )
        penalized_deviance = float(boot_result.deviance + pq_boot)
        if isinstance(distribution, Gaussian):
            assert likelihood_size is not None
            boot_scale = profile_gaussian_reml_scale(
                penalized_deviance,
                likelihood_size,
                M_p,
                saturated_log_weight=saturated_log_weight or 0.0,
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
        elif isinstance(distribution, Tweedie):
            assert tweedie_scale_data is not None
            boot_scale = profile_tweedie_reml_scale(
                tweedie_scale_data,
                penalized_deviance,
                M_p,
            )
            boot_phi = boot_scale.phi
            boot_inv_phi = boot_scale.inverse_phi
        else:
            boot_phi = max(
                penalized_deviance / max(_contract_size() - M_p, 1.0),
                1e-10,
            )
            boot_inv_phi = 1.0 / max(boot_phi, 1e-10)
    else:
        pq_boot = total_penalty_quadratic(
            boot_result.beta,
            boot_lambdas,
            penalties,
            list(dm.group_matrices),
        )
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
        beta_g = boot_result.beta[pc.group_sl]
        gm = dm.group_matrices[pc.group_index]
        quad = penalty_component_quadratic(pc, beta_g, gm)
        trace_term = as_hessian_factor(boot_inv).trace_inverse_penalty(pc)
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
                "omega_frob": (
                    float(np.sqrt(pc.group_sl.stop - pc.group_sl.start))
                    if pc.penalty_kind == "identity"
                    else float(np.linalg.norm(pc.omega_ssp))
                ),
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
    # Frozen directions from the previous iteration's active-set decision:
    # the stop criterion judges the ACTIVE set. An inferentially flat frozen
    # direction keeps a tiny persistent gradient forever (that is what makes
    # it flat), and counting it would spin the loop doing nothing until
    # max_reml_iter with every informative direction long determined.
    stop_criterion_frozen_d = None
    for poi_iter in range(max_reml_iter):
        rho_clipped = np.clip(rho, log_lo, log_hi)
        cand_lambdas = lambdas.copy()
        for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
            cand_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
        cand_lambdas.update(fixed_lambdas)

        # --- Step 1: One PIRLS step (W update) ---
        S_cand = (
            None
            if use_structured
            else build_penalty_matrix(
                dm.group_matrices,
                groups,
                cand_lambdas,
                p,
                reml_penalties=penalties,
            )
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
            reml_penalties=penalties,
            debug_recorder=debug_recorder,
            debug_context={"phase": "candidate", "reml_iteration": poi_iter + 1},
            trace_run=trace_run,
            trace_purpose="reml_candidate",
            weight_semantics=weight_semantics,
        )
        _t_pirls += _time.perf_counter() - _t0
        S_cand = latch_runtime_backend(
            pirls_result,
            cand_lambdas,
            S_cand,
            design=dm,
            penalty_components=penalties,
        )
        _n_pirls_steps += 1
        # The candidate is ONE working-model update. Its own convergence
        # flag says whether that update changed anything: when it did not,
        # the working model has settled at these lambdas and the next
        # iteration would recompute the same gradient, Hessian and step --
        # the discrete analogue of the exact engine's stationary mode.
        candidate_mode_stationary = bool(pirls_result.converged)
        warm_beta = pirls_result.beta.copy()
        warm_intercept = float(pirls_result.intercept)
        warm_deviance = float(pirls_result.deviance)

        c_centered_XtWX = cache.get("centered_XtWX")
        c_structured_system = cache.get("structured_system")
        if use_structured and not isinstance(
            c_structured_system,
            ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem,
        ):
            raise RuntimeError("Structured discrete REML cache is missing its block system.")
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
            saturated_log_weight=saturated_log_weight,
            weight_semantics=weight_semantics,
            gamma_scale_data=gamma_scale_data,
            tweedie_scale_data=tweedie_scale_data,
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
                        coefficient_width=p,
                    )
                if isinstance(objective_evaluation, REMLObjectiveEvaluation):
                    penalized_deviance = objective_evaluation.penalized_deviance
                else:
                    pq = total_penalty_quadratic(
                        pirls_result.beta,
                        cand_lambdas,
                        penalties,
                        list(dm.group_matrices),
                    )
                    penalized_deviance = float(pirls_result.deviance + pq)
                phi_hat = max(
                    penalized_deviance / max(_contract_size() - penalty_nullity, 1.0),
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

        if all_lambdas_fixed:
            rho = rho_clipped
            if profile is not None:
                # Fixed lambdas freeze definitionally -- the projection
                # zeroes their scores -- so the public freeze record exists
                # for this pre-Newton exit too. The zero derivatives are
                # the projection's definition, not measurements.
                profile["reml_freeze_decision"] = {
                    "names": list(group_names),
                    "proj_grad": [0.0] * m,
                    "hess_diag": [0.0] * m,
                    "row_curvature": [0.0] * m,
                    "penalty_rank": [
                        float(v) for v in direction_penalty_ranks(penalties, penalty_ranks)
                    ],
                    "normalized_curvature": [0.0] * m,
                    "curvature_bar": 0.0,
                    "score_scale": max(1.0 + abs(obj), 1.0),
                    "estimated": [bool(v) for v in estimated_mask],
                    "frozen": [True] * m,
                }
            converged = True
            termination_reason = "fixed_lambdas"
            break

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
        # The convergence decision belongs to this evaluated candidate.
        # Do not construct or install another Newton step before deciding
        # whether these exact lambdas are terminal.
        proj_grad_d = project_reml_gradient(
            grad,
            rho_clipped,
            estimated_mask,
            log_lower=log_lo,
            log_upper=log_hi,
        )
        stop_grad_d = mask_frozen_stop_gradient(
            proj_grad_d, stop_criterion_frozen_d, objective=obj, tolerance=_tol
        )
        candidate_convergence = evaluate_reml_candidate(
            iteration=poi_iter,
            objective=obj,
            previous_objective=prev_obj,
            projected_gradient=stop_grad_d,
            tolerance=_tol,
        )
        score_scale_d = candidate_convergence.score_scale
        proj_grad_norm = candidate_convergence.projected_gradient_norm
        obj_change = candidate_convergence.objective_change
        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            print(
                f"  POI iter {poi_iter + 1}  obj={obj:.4f}  "
                f"|grad|={proj_grad_norm:.6f}  delta_obj={obj_change:.6g}  [{lam_str}]"
            )
        revalidated_hess = None
        if candidate_convergence.converged:
            # Same curvature-only re-activation hole as the exact path: the
            # stop mask's gradient arm cannot see a masked direction whose
            # coupled partner raised its row curvature past the bar while
            # its gradient sits between reml_tol and the freeze floor.
            # Recompute the freeze decision against the current Hessian
            # before accepting; skipped when no masked direction holds a
            # gradient above the tolerance.
            masked_live = stop_criterion_frozen_d is not None and bool(
                np.any(
                    np.asarray(stop_criterion_frozen_d)
                    & (np.abs(proj_grad_d) >= _tol * score_scale_d)
                )
            )
            if not masked_live:
                rho = rho_clipped
                prev_obj = obj
                converged = True
                termination_reason = "score_objective_tolerance"
                _t_newton += _time.perf_counter() - _t0
                break
            revalidated_hess = reml_direct_hessian(
                dm.group_matrices,
                distribution,
                XtWX_S_inv,
                cand_lambdas,
                gradient=grad,
                penalty_caches=penalty_caches,
                pirls_result=pirls_result,
                n_obs=_contract_size(),
                phi_hat=phi_hat,
                inverse_phi=inverse_phi,
                d_inverse_phi_d_penalized_deviance=inverse_phi_derivative,
                penalty_nullity=penalty_nullity if not scale_known else None,
                reml_penalties=penalties,
                tensor_pair_evaluations=cand_tensor_pair_evals,
            )
            revalidation = freeze_flat_directions(
                proj_grad_d,
                revalidated_hess,
                direction_penalty_ranks(penalties, penalty_ranks),
                estimated_mask,
                objective=obj,
                tolerance=_tol,
            )
            if profile is not None:
                profile["reml_freeze_revalidated"] = True
                # The revalidation IS the last freeze decision made: when
                # it authorizes the exit below, the published record must
                # describe it, not iteration k-1.
                profile["reml_freeze_decision"] = {
                    "names": list(group_names),
                    "proj_grad": [float(abs(v)) for v in proj_grad_d],
                    "hess_diag": [float(revalidated_hess[i, i]) for i in range(m)],
                    "row_curvature": [float(v) for v in revalidation.row_curvature],
                    "penalty_rank": [float(v) for v in revalidation.penalty_rank],
                    "normalized_curvature": [float(v) for v in revalidation.normalized_curvature],
                    "curvature_bar": float(revalidation.curvature_bar),
                    "score_scale": float(score_scale_d),
                    "estimated": [bool(v) for v in estimated_mask],
                    "frozen": [bool(v) for v in revalidation.frozen],
                }
            reactivated = (
                ~np.asarray(revalidation.frozen)
                & np.asarray(stop_criterion_frozen_d)
                & (np.abs(proj_grad_d) >= _tol * score_scale_d)
            )
            if not bool(np.any(reactivated)):
                rho = rho_clipped
                prev_obj = obj
                converged = True
                termination_reason = "score_objective_tolerance"
                _t_newton += _time.perf_counter() - _t0
                break
            # Re-activated with a live gradient: keep iterating, reusing
            # the Hessian just computed.
        prev_obj = obj

        hess = (
            revalidated_hess
            if revalidated_hess is not None
            else reml_direct_hessian(
                dm.group_matrices,
                distribution,
                XtWX_S_inv,
                cand_lambdas,
                gradient=grad,
                penalty_caches=penalty_caches,
                pirls_result=pirls_result,
                n_obs=_contract_size(),
                phi_hat=phi_hat,
                inverse_phi=inverse_phi,
                d_inverse_phi_d_penalized_deviance=inverse_phi_derivative,
                penalty_nullity=penalty_nullity if not scale_known else None,
                reml_penalties=penalties,
                tensor_pair_evaluations=cand_tensor_pair_evals,
            )
        )

        # Active-set: freeze components with negligible gradient and Hessian.
        # The gradient/curvature bars live with the classifier's calibration
        # in reml/convergence.py. (The discrete engine's 1e-12 tolerance
        # floor once made a tolerance-coupled bar 1e-13 -- three decades
        # below where its own flat directions live; the shared floor and
        # curvature-relative arm close that.)
        direction_ranks = direction_penalty_ranks(penalties, penalty_ranks)
        freeze_decision = freeze_flat_directions(
            proj_grad_d,
            hess,
            direction_ranks,
            estimated_mask,
            objective=obj,
            tolerance=_tol,
        )
        frozen_d = freeze_decision.frozen
        active_idx_d = np.where(~frozen_d)[0]
        stop_criterion_frozen_d = frozen_d.copy()
        if profile is not None:
            # The freeze decision separates informative directions from
            # inferentially flat ones; the per-direction quantities it
            # judged are the calibration evidence for its bar. Overwritten
            # each iteration: what survives is the LAST decision MADE --
            # on a tolerance exit that is iteration k-1's, because the
            # freeze runs after the stop criterion that ended iteration k.
            profile["reml_freeze_decision"] = {
                "names": list(group_names),
                "proj_grad": [float(abs(v)) for v in proj_grad_d],
                "hess_diag": [float(hess[i, i]) for i in range(m)],
                "row_curvature": [float(v) for v in freeze_decision.row_curvature],
                "penalty_rank": [float(v) for v in freeze_decision.penalty_rank],
                "normalized_curvature": [float(v) for v in freeze_decision.normalized_curvature],
                "curvature_bar": float(freeze_decision.curvature_bar),
                "score_scale": float(score_scale_d),
                "estimated": [bool(v) for v in estimated_mask],
                "frozen": [bool(v) for v in frozen_d],
            }

        # Modified Newton: eigendecompose, flip negatives, floor small eigenvalues
        if active_idx_d.size == 0:
            rho = rho_clipped
            _t_newton += _time.perf_counter() - _t0
            if poi_iter == 0:
                # Do not let a deliberately loose tolerance bypass the
                # two-evaluation convergence contract.
                continue
            if not (obj_change < _tol * score_scale_d):
                # POI performs ONE working-model update per outer
                # iteration, so an early all-frozen verdict can be
                # measured at a nonstationary coefficient mode -- the
                # objective is still moving, and the next W update can
                # unfreeze what this one froze. All directions frozen
                # means rho stops moving, so continuing is a pure PIRLS
                # settle at the same lambdas; grant the exit only once
                # the objective arm agrees the mode has stabilized.
                continue
            converged = True
            termination_reason = "active_set_stationary"
            break

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
        trust_mu = 0.0
        trust_binding: list[str] = []
        gdot_newton = float(grad @ delta)
        gdot_damped = gdot_newton
        if use_tensor_surrogate_linesearch:
            # Bound log-lambda coordinates and each shared pair's difference
            # by damping the whole step. The coordinate bound also bounds
            # each pair's mean. Widen after one accepted full step so a
            # finite margin can move when its partner approaches infinity.
            # Frozen coordinates stay outside the active subspace.
            base_cap = 1.0 if not _tensor_post_stall_unlocked else 2.5
            cap_v = 0.25 if not _tensor_post_stall_unlocked else 1.0
            delta_newton = delta
            delta, trust_mu, trust_binding = _damped_tensor_newton_step(
                eigvecs_h,
                eigvals_pd,
                grad_sub_d,
                active_idx_d,
                m,
                group_names,
                shared_tensor_pairs,
                frozen_d,
                base_cap,
                cap_v,
            )
            gdot_damped = float(grad @ delta)
            # Descent holds by construction; the only admissible slack is
            # the round-off of the dot product itself. A violation means
            # the eigen floor or the active set is inconsistent with the
            # gradient, and continuing would hand the line search a step
            # it can never accept: name the state instead of stalling.
            descent_slack = 1e-12 * float(np.sum(np.abs(grad * delta)))
            if gdot_damped > descent_slack:
                raise RuntimeError(
                    "Damped discrete tensor step is not a descent direction: "
                    f"iteration {poi_iter + 1}, mu={trust_mu:.6g}, "
                    f"g.delta={gdot_damped:.6g} (undamped {gdot_newton:.6g}), "
                    f"binding={trust_binding}, active={active_idx_d.size}/{m}."
                )
            for group_name, (i, j) in shared_tensor_pairs:
                if frozen_d[i] or frozen_d[j]:
                    continue
                tensor_step_diag = {
                    "group_name": group_name,
                    "delta_u_raw": 0.5 * float(delta_newton[i] + delta_newton[j]),
                    "delta_u_used": 0.5 * float(delta[i] + delta[j]),
                    "delta_v_raw": 0.5 * float(delta_newton[i] - delta_newton[j]),
                    "delta_v_used": 0.5 * float(delta[i] - delta[j]),
                    "cap_u": base_cap,
                    "cap_v": cap_v,
                    "base_cap": base_cap,
                }
                break

        # Step capping on the generic path: scale the whole vector.
        if not use_tensor_surrogate_linesearch:
            local_max_newton_step = max_newton_step
            max_delta = float(np.max(np.abs(delta)))
            max_delta_raw = max_delta
            if max_delta > local_max_newton_step:
                delta *= local_max_newton_step / max_delta
        else:
            max_delta = float(np.max(np.abs(delta)))
            max_delta_raw = float(np.max(np.abs(delta_newton)))
        quad_grad = float(grad @ delta) if use_tensor_surrogate_linesearch else 0.0
        quad_curv = float(delta @ hess @ delta) if use_tensor_surrogate_linesearch else 0.0
        _t_newton += _time.perf_counter() - _t0
        _n_newton_steps += 1

        # --- Step 3: Line search (step halving on working-model REML) ---
        _t0 = _time.perf_counter()
        accepted = False
        step = 1.0
        halving_count = 0
        had_feasible_trial = False
        evaluated_feasible_trial = False
        first_full_eval_step: float | None = None
        # On the shared-tensor path the quadratic surrogate is a free
        # pre-filter on each step length, so the backtrack is not floored
        # (the old cap of five stopped at s = 1/16 and killed searches
        # whose model minimiser was smaller); the penalty build and the
        # true evaluation happen only for a length the surrogate lets
        # through, and a rejected true trial backtracks like the generic
        # path instead of ending the search.
        local_max_halving = max_halving
        if use_tensor_surrogate_linesearch and max_delta < 1e-12:
            local_max_halving = 0
        # H_pd - H is positive semidefinite. Along the damped direction,
        # positive curvature therefore places the quadratic minimizer at
        # s >= 1. No special interior step is needed during backtracking.
        for _ls in range(local_max_halving):
            rho_trial = np.clip(rho + step * delta, log_lo, log_hi)
            if use_tensor_surrogate_linesearch and bool(
                np.all(np.abs(rho_trial - rho_clipped) <= 1e-12)
            ):
                # Every moving coordinate is pinned at a bound; a shorter
                # step moves even less. Nothing feasible was tried.
                break
            had_feasible_trial = True
            trial_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_trial), strict=False):
                trial_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
            trial_lambdas.update(fixed_lambdas)

            _n_linesearch_evals += 1
            if use_tensor_surrogate_linesearch:
                _tls0 = _time.perf_counter()
                # The predicted CHANGE is judged, not obj + change: a
                # decrease below the objective's own resolution is still
                # a prediction, and the true evaluation it lets through is
                # the evidence a precision exit needs.
                predicted_change = step * quad_grad + 0.5 * (step**2) * quad_curv
                _t_linesearch_surrogate += _time.perf_counter() - _tls0
                _n_linesearch_surrogate_evals += 1
                if predicted_change >= 0.0:
                    step *= 0.5
                    halving_count += 1
                    continue
                _tfull0 = _time.perf_counter()

            S_trial = (
                None
                if use_structured
                else build_penalty_matrix(
                    dm.group_matrices,
                    groups,
                    trial_lambdas,
                    p,
                    reml_penalties=penalties,
                )
            )

            # Solve the cached profiled-intercept system analytically
            # (O(p^3), no data pass).
            _tls0 = _time.perf_counter()
            if use_structured:
                if not isinstance(
                    c_structured_system,
                    ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem,
                ):  # pragma: no cover - validated above
                    raise RuntimeError("Structured cached solve has no block system.")
                cached_solution = solve_cached_structured(
                    c_structured_system,
                    list(dm.group_matrices),
                    groups,
                    trial_lambdas,
                    reml_penalties=penalties,
                )
                beta_trial = cached_solution.beta
                intercept_trial = cached_solution.intercept
                log_det_H_trial = cached_solution.log_det_H
                hessian_rank_trial = cached_solution.hessian_rank
            else:
                if c_centered_XtWX is None or S_trial is None:
                    raise RuntimeError("Dense cached solve is missing matrix geometry.")
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
            cached_solve_elapsed = _time.perf_counter() - _tls0
            _t_linesearch_solve += cached_solve_elapsed
            if use_structured:
                _t_structured_cache_solve += cached_solve_elapsed
                _n_structured_cache_solves += 1
                if isinstance(
                    c_structured_system,
                    BlockStructuredSystem | SumToZeroBlockStructuredSystem,
                ):
                    _t_block_structured_cache_solve += cached_solve_elapsed
                    _n_block_structured_cache_solves += 1

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
                saturated_log_weight=saturated_log_weight,
                weight_semantics=weight_semantics,
                gamma_scale_data=gamma_scale_data,
                tweedie_scale_data=tweedie_scale_data,
            )
            _n_linesearch_full_evals += 1
            if use_tensor_surrogate_linesearch:
                _t_linesearch_full_obj += _time.perf_counter() - _tfull0
                if first_full_eval_step is None:
                    first_full_eval_step = float(step)
            # The cached trial solve is an exact solve of the profiled
            # working-model system, so the trial's own mode is stationary
            # by construction; a finite rejected objective is evidence.
            if trial_counts_as_precision_evidence(trial_pirls.converged, trial_obj):
                evaluated_feasible_trial = True

            if trial_obj < obj:
                rho = rho_trial
                warm_beta = beta_trial.copy()
                warm_intercept = intercept_trial
                warm_deviance = dev_trial
                accepted = True
                break

            step *= 0.5
            halving_count += 1

        _t_linesearch += _time.perf_counter() - _t0
        if use_tensor_surrogate_linesearch and accepted and halving_count == 0:
            _tensor_post_stall_unlocked = True

        if not accepted:
            # Every trial was rejected, so retain the evaluated candidate.
            # Installing an unscored fallback here can make a max-iteration
            # refit report lambdas that were never accepted by the criterion.
            rho = rho_clipped

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
                    "dead_search": bool(not accepted and had_feasible_trial),
                    "candidate_mode_stationary": candidate_mode_stationary,
                    "trust_mu": trust_mu,
                    "trust_binding": list(trust_binding),
                    "gdot_newton": gdot_newton,
                    "gdot_damped": gdot_damped,
                    "quad_grad": quad_grad,
                    "quad_curv": quad_curv,
                    "first_full_eval_step": first_full_eval_step,
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

        if use_tensor_surrogate_linesearch and not accepted and had_feasible_trial:
            # A dead line search on the shared-tensor path: every feasible
            # trial was rejected. The exit mirrors the exact engine's
            # (direct.py): the active gradient of the CURRENT active set --
            # the set this dead step actually moved -- is classified by
            # classify_dead_feasible_exit, which grants converged_at_precision
            # only when every active gradient is under the precision asked
            # for AND a true objective was evaluated and rejected, and
            # names an honest line_search_failed otherwise. Two things
            # differ from the exact engine, both because this engine's
            # candidate is ONE working-model update rather than a converged
            # PIRLS. First, the break waits for candidate_mode_stationary:
            # a dead search at an unsettled working model is not evidence
            # of a fixed point -- the gate measured a real additive binomial
            # fit whose every dead search (candidate PIRLS unconverged at
            # each) was followed by an accepted step once the next
            # working-model update moved the gradient at unchanged rho,
            # and breaking there published a different model, not the same
            # one sooner. A settled working model at unchanged rho means
            # the next iteration would recompute the same gradient, Hessian
            # and step and reject it again; on the measured synthetic stall
            # the flag turned true at the second dead search and the state
            # then repeated, digit for digit, for 27 iterations. Second, the
            # exit is confined to this path: the generic path keeps
            # iterating through a dead search, as measured, and its numbers
            # are untouched. A first-iteration exit is withheld like every
            # other converged exit here, so a loose tolerance cannot bypass
            # the two-evaluation contract.
            _n_dead_line_searches += 1
            if poi_iter > 0 and candidate_mode_stationary:
                active_grad_norm = (
                    float(np.max(np.abs(np.where(frozen_d, 0.0, proj_grad_d))))
                    if proj_grad_d.size
                    else 0.0
                )
                evidence = evaluated_feasible_trial and trial_counts_as_precision_evidence(
                    candidate_mode_stationary, obj
                )
                termination_reason = classify_dead_feasible_exit(
                    active_grad_norm,
                    objective=obj,
                    tolerance=_tol,
                    evaluated_trial=evidence,
                )
                converged = termination_reason == "converged_at_precision"
                if profile is not None:
                    profile["reml_dead_line_search"] = {
                        "iter": poi_iter + 1,
                        "active_gradient_norm": active_grad_norm,
                        "bar": float(max(FLAT_DIRECTION_FREEZE_FLOOR, _tol) * score_scale_d),
                        "evaluated_trial": bool(evidence),
                        "candidate_mode_stationary": bool(candidate_mode_stationary),
                        "termination_reason": termination_reason,
                    }
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
    S_final = (
        None
        if use_structured
        else build_penalty_matrix(
            dm.group_matrices,
            groups,
            final_lambdas,
            dm.p,
            reml_penalties=penalties,
        )
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
        reml_penalties=penalties,
        debug_recorder=debug_recorder,
        debug_context={"phase": "optimizer_final", "reml_iteration": poi_iter + 1},
        trace_run=trace_run,
        trace_purpose="reml_optimizer_final",
        weight_semantics=weight_semantics,
    )
    _t_pirls += _time.perf_counter() - _t0
    S_final = latch_runtime_backend(
        final_result,
        final_lambdas,
        S_final,
        design=dm,
        penalty_components=penalties,
    )
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
        saturated_log_weight=saturated_log_weight,
        weight_semantics=weight_semantics,
        gamma_scale_data=gamma_scale_data,
        tweedie_scale_data=tweedie_scale_data,
    )
    _t_objective += _time.perf_counter() - _t0
    # Always use the final refit -- it is the authoritative result from
    # full IRLS convergence at the converged lambdas.  The working-model
    # surrogates from the POI loop (n_iter=0) must not leak out.
    best_obj = final_obj
    best_lambdas = final_lambdas.copy()
    best_pirls = final_result
    if structured_runtime_fallback_reason is not None:
        best_pirls.direct_fallback_reason = structured_runtime_fallback_reason
    lambda_history.append(final_lambdas.copy())

    if profile is not None:
        if structured_runtime_fallback_reason is not None:
            profile["direct_fallback_reason"] = structured_runtime_fallback_reason
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
        profile["reml_structured_cache_solve_s"] = _t_structured_cache_solve
        profile["reml_n_structured_cache_solves"] = _n_structured_cache_solves
        profile["reml_block_structured_cache_solve_s"] = _t_block_structured_cache_solve
        profile["reml_n_block_structured_cache_solves"] = _n_block_structured_cache_solves
        # Lambda-only Schur re-solves consume cached O(q² + Kq) moments.
        # Objective evaluation may score eta separately, but the solve itself
        # never traverses row-scale design data.
        profile["reml_structured_cache_data_passes"] = 0
        profile["reml_block_structured_cache_data_passes"] = 0
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
        profile["reml_n_dead_line_searches"] = _n_dead_line_searches
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
        termination_reason=termination_reason,
        tweedie_scale_data=tweedie_scale_data,
        reml_penalties=penalties,
    )
