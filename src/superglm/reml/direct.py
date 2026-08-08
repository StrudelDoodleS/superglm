"""Direct REML Newton optimizer (exact path).

Damped Newton optimization of the REML/LAML objective with
W(rho) correction, gradient, Hessian, and Armijo line search.

References
----------
- Wood (2011) Section 6.2.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._fit_trace import TraceRun
from superglm.distributions import Gamma, Gaussian
from superglm.group_matrix import DesignMatrix
from superglm.reml.convergence import (
    classify_dead_feasible_exit,
    direction_penalty_ranks,
    evaluate_reml_candidate,
    freeze_flat_directions,
    mask_frozen_stop_gradient,
    project_reml_gradient,
)
from superglm.reml.discrete import optimize_discrete_reml_cached_w
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import (
    REMLObjectiveEvaluation,
    reml_laml_objective,
)
from superglm.reml.observed_geometry import (
    OBSERVED_PIRLS_TOL_CEILING,
    ObservedModeNotCertifiedError,
    ObservedModeNotConvergedError,
    ObservedREMLGeometry,
    build_observed_reml_geometry,
    classify_reml_curvature,
    mode_certification_hint,
    observed_mode_certification_bar,
    observed_penalized_mode_score,
    validate_observed_derivative_capability,
)
from superglm.reml.penalty_algebra import (
    build_penalty_matrix,
    coerce_reml_penalties,
    compute_penalty_nullity,
    penalty_component_quadratic,
    total_penalty_quadratic,
)
from superglm.reml.result import REMLResult
from superglm.reml.scale import (
    GammaScaleProfileData,
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
)
from superglm.reml.w_derivatives import reml_w_correction, validate_w_correction_order
from superglm.solvers.centered_system import TabmatCenteringState
from superglm.solvers.hessian_factor import as_hessian_factor
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.structured import resolve_structured_backend
from superglm.types import GroupSlice, PenaltyComponent

# The solve ceiling and the certification bar both live in observed_geometry
# now, shared with the terminal publication gate in model/reml_finalize.py so
# the two can never drift apart.
_OBSERVED_PIRLS_TOL_CEILING = OBSERVED_PIRLS_TOL_CEILING


def optimize_direct_reml(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    discrete: bool,
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
    max_analytical_per_w: int = 30,
    select_snap: bool = True,
    direct_solve: str = "auto",
    w_correction_order: int = 1,
    reml_penalties: list[PenaltyComponent] | None = None,
    estimated_names: set[str] | None = None,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
    debug_recorder=None,
    trace_run: TraceRun | None = None,
) -> REMLResult:
    """Optimize the direct REML objective via damped Newton (Wood 2011).

    Two algorithm variants depending on ``discrete``:

    **Exact path** (``discrete=False``):
        W(rho)-corrected direct REML with gradient, Hessian, line search.

    **Discrete path** (``discrete=True``):
        Cached-W fREML optimizer (fewer data passes), delegated to
        ``optimize_discrete_reml_cached_w``.  This deliberately uses the
        BAM-style cached working/Fisher curvature approximation and bypasses
        ordinary observed-Hessian LAML for noncanonical links.
    """
    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    w_correction_order = validate_w_correction_order(w_correction_order)
    if discrete:
        # The cached-W branch is the explicit BAM-style approximation boundary:
        # it does not claim Wood's exact observed-Hessian LAML geometry.
        return optimize_discrete_reml_cached_w(
            dm,
            distribution,
            link,
            groups,
            y,
            sample_weight,
            offset_arr,
            reml_groups,
            penalty_ranks,
            lambdas,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            verbose=verbose,
            penalty_caches=penalty_caches,
            profile=profile,
            max_analytical_per_w=max_analytical_per_w,
            select_snap=select_snap,
            direct_solve=direct_solve,
            reml_penalties=penalties,
            estimated_names=estimated_names,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
            debug_recorder=debug_recorder,
            trace_run=trace_run,
        )

    scale_known = getattr(distribution, "scale_known", True)
    likelihood_size: float | None = None
    gamma_scale_data: GammaScaleProfileData | None = None
    if isinstance(distribution, Gaussian):
        likelihood_size = float(np.sum(sample_weight, dtype=np.float64))
    elif isinstance(distribution, Gamma):
        gamma_scale_data = prepare_gamma_reml_scale_data(y, sample_weight)
    use_observed_geometry = (
        isinstance(dm, DesignMatrix)
        and classify_reml_curvature(
            distribution,
            link,
        )
        == "observed"
    )
    structured_decision = resolve_structured_backend(
        list(dm.group_matrices),
        groups,
        direct_solve=direct_solve,
        coefficient_width=dm.p,
        row_weights=sample_weight,
        lambda2=lambdas,
    )
    use_structured = structured_decision.use_structured
    if use_observed_geometry:
        validate_observed_derivative_capability(distribution, link, w_correction_order)
    observed_tabmat_state = TabmatCenteringState() if use_observed_geometry else None
    observed_pirls_tol = (
        min(pirls_tol, _OBSERVED_PIRLS_TOL_CEILING) if use_observed_geometry else pirls_tol
    )
    # Derived from the fixed ceiling, NOT from `observed_pirls_tol`. The bar
    # certifies that the mode is accurate enough to implicitly differentiate
    # through; that is a property of the derivative, not of how hard the caller
    # asked PIRLS to work. Scaling it with the caller's `tol` made the knob
    # self-defeating: the natural response to a certification failure is to
    # tighten the solve, which tightened the bar by the same factor, and
    # `tol=1e-14` could fail a fit that passed at every looser setting.
    observed_mode_tol = observed_mode_certification_bar()
    group_names = [pc.name for pc in penalties]
    m = len(group_names)
    # estimated_mask[i] = True  => component i is free to be optimized
    #                     False => component i has a fixed lambda (policy)
    if estimated_names is not None:
        estimated_mask = np.array([pc.name in estimated_names for pc in penalties])
    else:
        estimated_mask = np.ones(m, dtype=bool)
    log_lo, log_hi = np.log(1e-6), np.log(1e10)
    max_newton_step = 5.0
    _eps = np.finfo(float).eps
    _tol = reml_tol

    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    objective_history: list[float] = []
    warm_beta: NDArray | None = None
    warm_intercept: float | None = None

    best_obj = np.inf
    best_lambdas = lambdas.copy()
    best_pirls = None
    # In-loop candidate and trial fits skip per-fit rank metadata: the REML
    # gradient and objective read the O(p) geometry summary instead, and the
    # published statistics come from the terminal refit.  Diagnostic runs keep
    # full statistics so trace rows stay complete.
    loop_fit_statistics = trace_run is not None or debug_recorder is not None
    # Converged state of the accepted line-search trial, reusable as the next
    # iteration's candidate when the lambda signature is unchanged.
    _carry_forward: tuple | None = None
    converged = False
    termination_reason = "max_reml_iter"
    n_iter = 0
    all_lambdas_fixed = not bool(np.any(estimated_mask))

    _t_reml_start = _time.perf_counter()
    _t_pirls = 0.0
    _t_objective = 0.0
    _t_gradient = 0.0
    _t_hessian = 0.0
    _t_w_correction = 0.0
    _t_observed_geometry = 0.0
    _accepted_observed_mode_residual_max = 0.0
    _rejected_trial_observed_mode_residual_max = 0.0
    _observed_mode_rejected_trial_count = 0
    _t_linesearch = 0.0
    _n_linesearch_fits = 0
    structured_runtime_fallback_reason: str | None = None

    def latch_runtime_backend(
        pirls_result: PIRLSResult,
        lambda_values: dict[str, float],
        penalty: NDArray | None,
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
            list(dm.group_matrices),
            groups,
            lambda_values,
            dm.p,
            reml_penalties=penalties,
        )

    # === Bootstrap: one FP step from conservative interaction penalties ===
    # Rich tensor interactions can explode under an almost-unpenalized
    # bootstrap fit. Keep main-effect bootstrap lambdas tiny, but start
    # interaction penalty components from a materially stronger seed.
    boot_lambdas = {pc.name: (1.0 if ":" in pc.group_name else 1e-4) for pc in penalties}
    S_boot = (
        None
        if use_structured
        else build_penalty_matrix(
            dm.group_matrices,
            groups,
            boot_lambdas,
            dm.p,
            reml_penalties=penalties,
        )
    )
    _t0 = _time.perf_counter()
    boot_result, boot_inv, boot_xtwx = fit_irls_direct(
        X=dm,
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
        profile=profile,
        direct_solve=direct_solve,
        S_override=S_boot,
        reml_penalties=penalties,
        debug_recorder=debug_recorder,
        debug_context={"phase": "bootstrap", "reml_iteration": 0},
        trace_run=trace_run,
        trace_purpose="reml_bootstrap",
    )
    _t_pirls += _time.perf_counter() - _t0
    S_boot = latch_runtime_backend(boot_result, boot_lambdas, S_boot)
    warm_beta = boot_result.beta.copy()
    warm_intercept = float(boot_result.intercept)

    boot_phi = 1.0
    boot_inv_phi = 1.0
    if not scale_known:
        pq_boot = total_penalty_quadratic(
            boot_result.beta,
            boot_lambdas,
            penalties,
            dm.group_matrices,
        )
        boot_hessian_rank = boot_result.reml_hessian_rank
        if boot_hessian_rank is None:
            boot_hessian_rank = 1 + dm.p
        boot_penalty_nullity = compute_penalty_nullity(
            S_boot,
            hessian_rank=boot_hessian_rank,
            penalties=penalties,
            lambdas=boot_lambdas,
            coefficient_width=dm.p,
        )
        penalized_deviance = float(boot_result.deviance + pq_boot)
        if isinstance(distribution, Gaussian):
            assert likelihood_size is not None
            boot_scale = profile_gaussian_reml_scale(
                penalized_deviance,
                likelihood_size,
                boot_penalty_nullity,
            )
            boot_phi = boot_scale.phi
            boot_inv_phi = boot_scale.inverse_phi
        elif isinstance(distribution, Gamma):
            assert gamma_scale_data is not None
            boot_scale = profile_gamma_reml_scale(
                gamma_scale_data,
                penalized_deviance,
                boot_penalty_nullity,
            )
            boot_phi = boot_scale.phi
            boot_inv_phi = boot_scale.inverse_phi
        else:
            boot_phi = max(
                penalized_deviance / max(len(y) - boot_penalty_nullity, 1.0),
                1e-10,
            )
            boot_inv_phi = 1.0 / max(boot_phi, 1e-10)
    bootstrap_log_step_cap = 4.0

    # Store original fixed lambda values so they can be restored exactly
    # after the exp(rho)->clip round-trip (which would clamp 0.0 to 1e-6).
    fixed_lambdas: dict[str, float] = {}
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_lambdas[pc.name] = float(lambdas[pc.name])

    rho = np.zeros(m, dtype=np.float64)
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_val = fixed_lambdas[pc.name]
            rho[i] = np.clip(np.log(max(fixed_val, 1e-6)), log_lo, log_hi)
            continue
        gm = dm.group_matrices[pc.group_index]
        beta_g = boot_result.beta[pc.group_sl]
        quad = penalty_component_quadratic(pc, beta_g, gm)
        trace_term = as_hessian_factor(boot_inv).trace_inverse_penalty(pc)
        r_j = pc.rank if pc.rank > 0 else (penalty_ranks[pc.name] if penalty_ranks else 0.0)
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        # Snap degenerate select=True null-space penalties to upper bound.
        # When quad << trace, the FP update is degenerate
        # (any lambda is approx a fixed point).  Snap breaks it.
        if (
            select_snap
            and pc.component_type == "selection"
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

    prev_obj = np.inf
    # Frozen directions from the previous iteration's active-set decision:
    # the stop criterion judges the ACTIVE set. An inferentially flat frozen
    # direction keeps a tiny persistent gradient forever (that is what makes
    # it flat), and counting it would spin the loop doing nothing until
    # max_reml_iter with every informative direction long determined.
    stop_criterion_frozen = None

    if verbose:
        boot_lam_str = ", ".join(
            f"{name}={np.exp(rho[i]):.4g}" for i, name in enumerate(group_names)
        )
        print(f"  REML bootstrap: lambdas=[{boot_lam_str}]")

    for outer in range(max_reml_iter):
        n_iter = outer + 1
        rho_clipped = np.clip(rho, log_lo, log_hi)

        cand_lambdas = lambdas.copy()
        for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
            cand_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
        # Restore exact fixed values (exp->clip would clamp 0.0 to 1e-6)
        cand_lambdas.update(fixed_lambdas)

        if _carry_forward is not None and _carry_forward[0] == cand_lambdas:
            _, pirls_result, XtWX_S_inv, XtWX, S_cand = _carry_forward
            if profile is not None:
                profile["reml_candidate_reuses"] = profile.get("reml_candidate_reuses", 0) + 1
        else:
            S_cand = (
                None
                if use_structured
                else build_penalty_matrix(
                    dm.group_matrices,
                    groups,
                    cand_lambdas,
                    dm.p,
                    reml_penalties=penalties,
                )
            )
            _t0 = _time.perf_counter()
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
                max_iter=max_pirls_iter,
                tol=observed_pirls_tol,
                convergence="coefficients" if use_observed_geometry else "deviance",
                return_xtwx=True,
                profile=profile,
                direct_solve=direct_solve,
                S_override=S_cand,
                reml_penalties=penalties,
                compute_rank_info=loop_fit_statistics,
                _compute_fit_statistics=loop_fit_statistics,
                debug_recorder=debug_recorder,
                debug_context={"phase": "candidate", "reml_iteration": n_iter},
                trace_run=trace_run,
                trace_purpose="reml_candidate",
            )
            _t_pirls += _time.perf_counter() - _t0
        S_cand = latch_runtime_backend(pirls_result, cand_lambdas, S_cand)
        warm_beta = pirls_result.beta.copy()
        warm_intercept = float(pirls_result.intercept)

        geometry: ObservedREMLGeometry | None = None
        reml_inverse = XtWX_S_inv
        objective_logdet = pirls_result.log_det_H
        objective_hessian_rank: int | None = None
        if use_observed_geometry:
            if not pirls_result.converged:
                # Typed so a power search can score this point infeasible and
                # route around it, exactly as it does the certification
                # failure below -- same physical condition, earlier door.
                raise ObservedModeNotConvergedError(hint=mode_certification_hint(distribution))
            _t0 = _time.perf_counter()
            geometry = build_observed_reml_geometry(
                dm=dm,
                distribution=distribution,
                link=link,
                y=y,
                sample_weight=sample_weight,
                offset_arr=offset_arr,
                result=pirls_result,
                penalty=S_cand,
                tabmat_state=observed_tabmat_state,
                derivative_order=w_correction_order,
                groups=groups if use_structured else None,
                lambdas=cand_lambdas if use_structured else None,
                reml_penalties=penalties if use_structured else None,
                structured_group_index=(
                    structured_decision.group_index if use_structured else None
                ),
            )
            _t_observed_geometry += _time.perf_counter() - _t0
            if geometry.hessian_inverse is None:  # pragma: no cover - requested above
                raise RuntimeError("observed REML geometry omitted its slope inverse")
            reml_inverse = geometry.hessian_inverse
            objective_logdet = geometry.log_det_H
            objective_hessian_rank = geometry.hessian_rank
            mode_score = observed_penalized_mode_score(
                dm=dm,
                distribution=distribution,
                link=link,
                y=y,
                sample_weight=sample_weight,
                result=pirls_result,
                penalty=S_cand,
                geometry=geometry,
                lambdas=cand_lambdas if use_structured else None,
                reml_penalties=penalties if use_structured else None,
            )
            _accepted_observed_mode_residual_max = max(
                _accepted_observed_mode_residual_max,
                mode_score.relative_max,
            )
            if mode_score.relative_max > observed_mode_tol:
                raise ObservedModeNotCertifiedError(
                    mode_score.relative_max,
                    observed_mode_tol,
                    hint=mode_certification_hint(distribution),
                )

        _t0 = _time.perf_counter()
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
            log_det_H=objective_logdet,
            hessian_rank=objective_hessian_rank,
            S_override=S_cand,
            reml_penalties=penalties,
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
            # Compatibility for lightweight test/instrumentation callbacks
            # that replace the public objective with a scalar-returning stub.
            obj = float(objective_evaluation)
            profiled_scale = None
        if not scale_known:
            if profiled_scale is not None:
                phi_hat = profiled_scale.phi
                inverse_phi = profiled_scale.inverse_phi
                inverse_phi_derivative = profiled_scale.d_inverse_phi_d_penalized_deviance
            else:
                if penalty_nullity is None:
                    hessian_rank = pirls_result.reml_hessian_rank
                    if hessian_rank is None:
                        hessian_rank = 1 + dm.p
                    penalty_nullity = compute_penalty_nullity(
                        S_cand,
                        hessian_rank=hessian_rank,
                        penalties=penalties,
                        lambdas=cand_lambdas,
                        coefficient_width=dm.p,
                    )
                if isinstance(objective_evaluation, REMLObjectiveEvaluation):
                    penalized_deviance = objective_evaluation.penalized_deviance
                else:
                    pq = total_penalty_quadratic(
                        pirls_result.beta,
                        cand_lambdas,
                        penalties,
                        dm.group_matrices,
                    )
                    penalized_deviance = float(pirls_result.deviance + pq)
                phi_hat = max(
                    penalized_deviance / max(len(y) - penalty_nullity, 1.0),
                    1e-10,
                )
                inverse_phi = 1.0 / max(phi_hat, 1e-10)
        _t_objective += _time.perf_counter() - _t0
        objective_history.append(float(obj))
        if trace_run is not None and trace_run.enabled:
            if pirls_result.state_id is None:  # pragma: no cover - trace contract
                raise RuntimeError("traced REML candidate is missing its coefficient state ID")
            trace_run.emit_lazy(
                "evaluation",
                lambda: {
                    "state_id": pirls_result.state_id,
                    "evaluation_id": pirls_result.evaluation_id,
                    "solver": "direct_reml",
                    "phase": "candidate",
                    "outer_iteration": n_iter,
                    "objective": float(obj),
                    "lambdas": cand_lambdas,
                    "dispersion": float(phi_hat),
                    "effective_df": float(pirls_result.effective_df),
                },
                channel="reml",
                purpose="reml_candidate",
                authoritative=False,
            )

        if all_lambdas_fixed:
            best_obj = obj
            best_lambdas = cand_lambdas.copy()
            best_pirls = pirls_result
            lambda_history.append(cand_lambdas.copy())
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
                    "frozen": [True] * m,
                }
            converged = True
            termination_reason = "fixed_lambdas"
            break

        _t0 = _time.perf_counter()
        grad_partial = reml_direct_gradient(
            dm.group_matrices,
            pirls_result,
            reml_inverse,
            cand_lambdas,
            reml_penalties=penalties,
            phi_hat=phi_hat,
            inverse_phi=inverse_phi,
        )
        _t_gradient += _time.perf_counter() - _t0

        # W(rho) correction
        _t0 = _time.perf_counter()
        if not discrete:
            w_corr = reml_w_correction(
                dm,
                link,
                groups,
                pirls_result,
                reml_inverse,
                cand_lambdas,
                penalty_caches=penalty_caches,
                sample_weight=sample_weight,
                offset_arr=offset_arr,
                distribution=distribution,
                w_correction_order=w_correction_order,
                reml_penalties=penalties,
                geometry=geometry,
            )
        else:
            w_corr = None
        _t_w_correction += _time.perf_counter() - _t0
        if w_corr is not None:
            grad_w_correction = w_corr[0]
            dH_extra = w_corr[1]
            dH2_cross = w_corr[2] if len(w_corr) > 2 else None
            grad = grad_partial + grad_w_correction
        else:
            grad = grad_partial.copy()
            dH_extra = None
            dH2_cross = None

        if obj < best_obj:
            best_obj = obj
            best_lambdas = cand_lambdas.copy()
            best_pirls = pirls_result

        lambda_history.append(cand_lambdas.copy())

        proj_grad = project_reml_gradient(
            grad,
            rho_clipped,
            estimated_mask,
            log_lower=log_lo,
            log_upper=log_hi,
        )
        stop_grad = mask_frozen_stop_gradient(
            proj_grad, stop_criterion_frozen, objective=obj, tolerance=_tol
        )
        candidate_convergence = evaluate_reml_candidate(
            iteration=outer,
            objective=obj,
            previous_objective=prev_obj,
            projected_gradient=stop_grad,
            tolerance=_tol,
        )
        proj_grad_norm = candidate_convergence.projected_gradient_norm
        score_scale = candidate_convergence.score_scale
        obj_change = candidate_convergence.objective_change

        if verbose:
            lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
            print(
                f"  REML Newton iter={n_iter}  obj={obj:.4f}  "
                f"|grad|={proj_grad_norm:.6f}  delta_obj={obj_change:.6g}  "
                f"lambdas=[{lam_str}]"
            )

        prev_obj = obj

        if candidate_convergence.converged:
            converged = True
            termination_reason = "score_objective_tolerance"
            break

        # Wood outer-Hessian update.  With ``w_correction_order=2`` this
        # includes the exact available second curvature derivatives; the
        # default order 1 is an exact objective/gradient with a modified
        # (quasi-Newton) Hessian.  Wood (2011) eq. 6.2 writes the
        # diagonal dS_i/dρ_i term as g_i + 0.5*r_i, where g_i is the fixed-W
        # gradient.  The W(rho) terms are differentiated separately through
        # dH_extra and dH2_cross; using the total gradient here would add the
        # first-order W correction twice on the diagonal.
        _t0 = _time.perf_counter()
        hess = reml_direct_hessian(
            dm.group_matrices,
            distribution,
            reml_inverse,
            cand_lambdas,
            gradient=grad_partial,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
            inverse_phi=inverse_phi,
            d_inverse_phi_d_penalized_deviance=inverse_phi_derivative,
            penalty_nullity=penalty_nullity if not scale_known else None,
            dH_extra=dH_extra,
            dH2_cross=dH2_cross,
            reml_penalties=penalties,
        )

        # Active-set: freeze components with negligible gradient and Hessian.
        # The gradient/curvature bars live with the classifier's calibration
        # in reml/convergence.py.
        direction_ranks = direction_penalty_ranks(penalties, penalty_ranks)
        freeze_decision = freeze_flat_directions(
            proj_grad,
            hess,
            direction_ranks,
            estimated_mask,
            objective=obj,
            tolerance=_tol,
        )
        frozen = freeze_decision.frozen
        active_idx = np.where(~frozen)[0]
        stop_criterion_frozen = frozen.copy()
        if profile is not None:
            # The freeze decision separates informative directions from
            # inferentially flat ones; the per-direction quantities it
            # judged are the calibration evidence for its bar. Overwritten
            # each iteration: what survives is the LAST decision MADE --
            # on a tolerance exit that is iteration k-1's, because the
            # freeze runs after the stop criterion that ended iteration k.
            profile["reml_freeze_decision"] = {
                "names": list(group_names),
                "proj_grad": [float(abs(v)) for v in proj_grad],
                "hess_diag": [float(hess[i, i]) for i in range(m)],
                "row_curvature": [float(v) for v in freeze_decision.row_curvature],
                "penalty_rank": [float(v) for v in freeze_decision.penalty_rank],
                "normalized_curvature": [float(v) for v in freeze_decision.normalized_curvature],
                "curvature_bar": float(freeze_decision.curvature_bar),
                "score_scale": float(score_scale),
                "frozen": [bool(v) for v in frozen],
            }

        if active_idx.size == 0:
            rho = rho_clipped
            _t_hessian += _time.perf_counter() - _t0
            if outer == 0:
                # Do not let a deliberately loose tolerance bypass the
                # two-evaluation convergence contract.
                continue
            # All components frozen -- converged. Usually the compound
            # criterion fires first (an all-frozen mask zeroes the next
            # iteration's stop gradient), so this exit needs every
            # direction to cross the freeze bar in the same iteration
            # that the objective is still moving -- a genuinely all-flat
            # model, not the common endgame.
            converged = True
            termination_reason = "active_set_stationary"
            break

        # Modified Newton: eigendecompose, flip negatives, floor small eigenvalues
        if active_idx.size < m:
            hess_sub = hess[np.ix_(active_idx, active_idx)]
            grad_sub = grad[active_idx]
        else:
            hess_sub = hess
            grad_sub = grad

        eigvals_h, eigvecs_h = np.linalg.eigh(hess_sub)
        max_eig = max(abs(eigvals_h).max(), 1e-12)
        eig_floor = max_eig * _eps**0.7
        eigvals_pd = np.maximum(np.abs(eigvals_h), eig_floor)

        hess_pd = (eigvecs_h * eigvals_pd) @ eigvecs_h.T
        delta_sub = -np.linalg.solve(hess_pd, grad_sub)

        # Scatter back to full delta
        delta = np.zeros(m)
        delta[active_idx] = delta_sub

        # Proportional step cap: scale entire vector if any component > max_step
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_newton_step:
            delta *= max_newton_step / max_delta
        _t_hessian += _time.perf_counter() - _t0

        # Step-halving line search with Armijo condition
        _t0 = _time.perf_counter()
        max_ls = 8
        _carry_forward = None
        step = 1.0
        armijo_c = 1e-4
        descent = float(grad @ delta)
        accepted = False
        had_feasible_trial = False
        evaluated_feasible_trial = False
        for _ls in range(max_ls):
            rho_trial = np.clip(rho_clipped + step * delta, log_lo, log_hi)
            if np.all(np.abs(rho_trial - rho_clipped) <= 1e-12):
                break
            had_feasible_trial = True
            trial_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_trial), strict=False):
                trial_lambdas[name] = float(np.clip(val, 1e-6, 1e10))
            trial_lambdas.update(fixed_lambdas)

            _n_linesearch_fits += 1
            S_trial = (
                None
                if use_structured
                else build_penalty_matrix(
                    dm.group_matrices,
                    groups,
                    trial_lambdas,
                    dm.p,
                    reml_penalties=penalties,
                )
            )
            trial_result, trial_inv, trial_xtwx = fit_irls_direct(
                X=dm,
                y=y,
                weights=sample_weight,
                family=distribution,
                link=link,
                groups=groups,
                lambda2=trial_lambdas,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                max_iter=max_pirls_iter,
                tol=observed_pirls_tol,
                convergence="coefficients" if use_observed_geometry else "deviance",
                return_xtwx=True,
                profile=profile,
                direct_solve=direct_solve,
                S_override=S_trial,
                reml_penalties=penalties,
                compute_rank_info=loop_fit_statistics,
                _compute_fit_statistics=loop_fit_statistics,
                debug_recorder=debug_recorder,
                debug_context={
                    "phase": "line_search",
                    "reml_iteration": n_iter,
                    "line_search_iteration": _ls + 1,
                    "trial_alpha": float(step),
                },
                trace_run=trace_run,
                trace_purpose="reml_line_search",
            )
            S_trial = latch_runtime_backend(trial_result, trial_lambdas, S_trial)

            trial_logdet = trial_result.log_det_H
            trial_hessian_rank: int | None = None
            trial_mode_residual: float | None = None
            if use_observed_geometry:
                if not trial_result.converged:
                    step *= 0.5
                    continue
                _t_geometry = _time.perf_counter()
                try:
                    trial_geometry = build_observed_reml_geometry(
                        dm=dm,
                        distribution=distribution,
                        link=link,
                        y=y,
                        sample_weight=sample_weight,
                        offset_arr=offset_arr,
                        result=trial_result,
                        penalty=S_trial,
                        tabmat_state=observed_tabmat_state,
                        compute_inverse=False,
                        groups=groups if use_structured else None,
                        lambdas=trial_lambdas if use_structured else None,
                        reml_penalties=penalties if use_structured else None,
                        structured_group_index=(
                            structured_decision.group_index if use_structured else None
                        ),
                    )
                except ValueError:
                    _t_observed_geometry += _time.perf_counter() - _t_geometry
                    _observed_mode_rejected_trial_count += 1
                    step *= 0.5
                    continue
                _t_observed_geometry += _time.perf_counter() - _t_geometry
                trial_logdet = trial_geometry.log_det_H
                trial_hessian_rank = trial_geometry.hessian_rank
                trial_mode_score = observed_penalized_mode_score(
                    dm=dm,
                    distribution=distribution,
                    link=link,
                    y=y,
                    sample_weight=sample_weight,
                    result=trial_result,
                    penalty=S_trial,
                    geometry=trial_geometry,
                    lambdas=trial_lambdas if use_structured else None,
                    reml_penalties=penalties if use_structured else None,
                )
                trial_mode_residual = trial_mode_score.relative_max
                if trial_mode_score.relative_max > observed_mode_tol:
                    _observed_mode_rejected_trial_count += 1
                    _rejected_trial_observed_mode_residual_max = max(
                        _rejected_trial_observed_mode_residual_max,
                        trial_mode_score.relative_max,
                    )
                    step *= 0.5
                    continue

            trial_evaluation = reml_laml_objective(
                dm,
                distribution,
                link,
                groups,
                y,
                trial_result,
                trial_lambdas,
                sample_weight,
                offset_arr,
                XtWX=trial_xtwx,
                penalty_caches=penalty_caches,
                log_det_H=trial_logdet,
                hessian_rank=trial_hessian_rank,
                S_override=S_trial,
                reml_penalties=penalties,
                likelihood_size=likelihood_size,
                gamma_scale_data=gamma_scale_data,
                return_evaluation=True,
            )
            trial_obj = (
                trial_evaluation.value
                if isinstance(trial_evaluation, REMLObjectiveEvaluation)
                else float(trial_evaluation)
            )
            evaluated_feasible_trial = True

            armijo_bound = obj + armijo_c * step * descent
            trial_accepted = bool(trial_obj <= armijo_bound)
            if trace_run is not None and trace_run.enabled:
                if trial_result.state_id is None:  # pragma: no cover - trace contract
                    raise RuntimeError("traced REML line-search trial is missing its state ID")
                trace_run.emit_lazy(
                    "evaluation",
                    lambda: {
                        "state_id": trial_result.state_id,
                        "evaluation_id": trial_result.evaluation_id,
                        "solver": "direct_reml",
                        "phase": "line_search",
                        "outer_iteration": n_iter,
                        "line_search_iteration": _ls + 1,
                        "trial_alpha": float(step),
                        "objective": float(trial_obj),
                        "armijo_bound": float(armijo_bound),
                        "accepted": trial_accepted,
                        "lambdas": trial_lambdas,
                    },
                    channel="reml",
                    purpose="reml_line_search",
                    authoritative=False,
                )

            if trial_accepted:
                if trial_mode_residual is not None:
                    _accepted_observed_mode_residual_max = max(
                        _accepted_observed_mode_residual_max,
                        trial_mode_residual,
                    )
                rho = rho_trial
                warm_beta = trial_result.beta.copy()
                warm_intercept = float(trial_result.intercept)
                # The accepted line-search state has already paid for a full
                # PIRLS solve and objective evaluation.  It is therefore a
                # valid retained candidate even when this is the final outer
                # iteration; waiting until the next loop would silently
                # discard a guaranteed improvement at ``max_reml_iter``.
                if trial_obj < best_obj:
                    best_obj = float(trial_obj)
                    best_lambdas = trial_lambdas.copy()
                    best_pirls = trial_result
                    if outer == max_reml_iter - 1:
                        lambda_history.append(trial_lambdas.copy())
                        objective_history.append(float(trial_obj))
                # The accepted trial's lambdas become the next iteration's
                # candidate lambdas, and the candidate refit at the top of the
                # loop solves the identical penalised system. Carry the
                # converged state forward instead of recomputing it — but only
                # a CONVERGED state: an Armijo-accepted trial that exhausted
                # max_pirls_iter is nonstationary, and reusing it would put
                # the next gradient/Hessian at the wrong coefficients instead
                # of warm-start refitting them.
                if trial_result.converged:
                    _carry_forward = (
                        trial_lambdas.copy(),
                        trial_result,
                        trial_inv,
                        trial_xtwx,
                        S_trial,
                    )
                else:
                    _carry_forward = None
                accepted = True
                break
            if trial_mode_residual is not None:
                _observed_mode_rejected_trial_count += 1
                _rejected_trial_observed_mode_residual_max = max(
                    _rejected_trial_observed_mode_residual_max,
                    trial_mode_residual,
                )
            step *= 0.5

        if not accepted:
            rho = rho_clipped
            _t_linesearch += _time.perf_counter() - _t0
            if not had_feasible_trial:
                # All projected coordinates are stationary at a bound.
                # Re-evaluate this same point once so the compound objective
                # criterion can confirm stability without redundant trial fits.
                continue
            # The fully converged exact objective rejected every feasible
            # trial. If every direction in the CURRENT active set -- the set
            # this dead step actually moved -- has its gradient below the
            # precision actually asked for (the resolved tolerance, never
            # tighter than the achievable-precision floor), the optimum is
            # resolved and the dead line search is the proof, not a
            # failure: the predicted decrease from a sub-bar gradient is
            # quadratically below candidate-machinery noise, so the last
            # decades of an ultra-tight bar do not exist to be closed. The
            # objective's own history is no veto here: the last accepted
            # step belongs to a since-frozen flat direction whenever one
            # just crossed the freeze bar, and a dead feasible search
            # already states that no further objective progress exists. A
            # genuinely undetermined stall keeps its active gradient orders
            # above the bar and stays an honest failure.
            active_grad_norm = (
                float(np.max(np.abs(np.where(frozen, 0.0, proj_grad)))) if proj_grad.size else 0.0
            )
            termination_reason = classify_dead_feasible_exit(
                active_grad_norm,
                objective=obj,
                tolerance=_tol,
                evaluated_trial=evaluated_feasible_trial,
            )
            converged = termination_reason == "converged_at_precision"
            break
        _t_linesearch += _time.perf_counter() - _t0

    if best_pirls is None:
        raise RuntimeError("Direct REML Newton did not evaluate any candidates")
    if structured_runtime_fallback_reason is not None:
        best_pirls.direct_fallback_reason = structured_runtime_fallback_reason

    if profile is not None:
        if structured_runtime_fallback_reason is not None:
            profile["direct_fallback_reason"] = structured_runtime_fallback_reason
        profile["reml_optimizer_s"] = _time.perf_counter() - _t_reml_start
        profile["reml_pirls_s"] = _t_pirls
        profile["reml_objective_s"] = _t_objective
        profile["reml_gradient_s"] = _t_gradient
        profile["reml_w_correction_s"] = _t_w_correction
        profile["reml_observed_geometry_s"] = _t_observed_geometry
        profile["reml_observed_mode_residual_accepted_max"] = _accepted_observed_mode_residual_max
        profile["reml_observed_mode_residual_rejected_trial_max"] = (
            _rejected_trial_observed_mode_residual_max
        )
        profile["reml_observed_mode_rejected_trial_count"] = _observed_mode_rejected_trial_count
        profile["reml_w_correction_order"] = int(w_correction_order)
        profile["reml_hessian_newton_s"] = _t_hessian
        profile["reml_linesearch_s"] = _t_linesearch
        profile["reml_n_linesearch_fits"] = _n_linesearch_fits
        profile["reml_n_outer_iter"] = n_iter

    return REMLResult(
        lambdas=best_lambdas,
        pirls_result=best_pirls,
        n_reml_iter=n_iter,
        converged=converged,
        lambda_history=lambda_history,
        objective=float(best_obj),
        objective_history=objective_history,
        curvature_source="observed" if use_observed_geometry else "fisher",
        termination_reason=termination_reason,
    )
