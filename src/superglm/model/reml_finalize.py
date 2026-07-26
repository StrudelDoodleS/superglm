"""Internal REML fit finalization helpers."""

from __future__ import annotations

import time as _time
from dataclasses import replace

import numpy as np

from superglm._fit_trace import TraceRun
from superglm._reporting_state import (
    FactorSmoothLevelSupport,
    StructuredLevelSupport,
    build_reporting_support_state,
)
from superglm.distributions import Gamma, Gaussian, Tweedie, clip_mu
from superglm.links import stabilize_eta
from superglm.model.base import rebuild_dm_with_lambdas
from superglm.model.reml_state import update_reml_r_inv
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.observed_geometry import (
    build_observed_reml_geometry,
    classify_reml_curvature,
    observed_penalized_mode_score,
)
from superglm.reml.penalty_algebra import (
    build_penalty_context,
    build_penalty_matrix,
    build_tensor_pair_logdet_summaries,
    compute_penalty_nullity,
    evaluate_tensor_pair_logdet_summaries,
    total_penalty_quadratic,
)
from superglm.reml.result import _map_beta_between_bases
from superglm.reml.scale import (
    GammaScaleProfileData,
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.structured import (
    BlockSchurFactor,
    BlockStructuredSystem,
    BlockSymmetricOperator,
    CenteredBlockOperator,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
    ScalarStructuredSystem,
    StructuredLinearSystemState,
    SumToZeroBlockOperator,
    SumToZeroBlockStructuredSystem,
    SymmetricBlockOperator,
)
from superglm.solvers.sum_to_zero import (
    ProfiledSumToZeroBlockFactor,
    SumToZeroBlockFactor,
)


def _build_structured_linear_system_state(
    *,
    factor,
    data_operator,
    cache: dict,
    support_totals: dict[
        str,
        StructuredLevelSupport | FactorSmoothLevelSupport,
    ],
) -> StructuredLinearSystemState | None:
    """Distill a final structured refit into compact persistent state."""
    if not isinstance(
        factor,
        ProfiledScalarSchurFactor | ProfiledBlockSchurFactor | ProfiledSumToZeroBlockFactor,
    ):
        return None
    system = cache.get("structured_system")
    penalized_operator = cache.get("penalized_operator")
    if not isinstance(
        system,
        ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem,
    ) or not isinstance(
        penalized_operator,
        SymmetricBlockOperator | BlockSymmetricOperator | SumToZeroBlockOperator,
    ):
        raise RuntimeError("terminal structured refit omitted its compact system state")
    if not isinstance(
        data_operator,
        SymmetricBlockOperator | BlockSymmetricOperator | SumToZeroBlockOperator,
    ):
        raise RuntimeError("terminal structured refit omitted its compact data operator")

    if isinstance(penalized_operator, SumToZeroBlockOperator):
        coefficient_factor = SumToZeroBlockFactor(
            A=penalized_operator.A,
            C=penalized_operator.C,
            D=penalized_operator.D,
            small_indices=penalized_operator.small_indices,
            structured_indices=penalized_operator.structured_indices,
            term_name=system.dominant_group_name,
            level_labels=system.level_labels,
        )
    elif isinstance(penalized_operator, BlockSymmetricOperator):
        coefficient_factor = BlockSchurFactor(
            A=penalized_operator.A,
            C=penalized_operator.C,
            D=penalized_operator.D,
            small_indices=penalized_operator.small_indices,
            structured_indices=penalized_operator.structured_indices,
            term_name=system.dominant_group_name,
        )
    else:
        coefficient_factor = ScalarSchurFactor(
            A=penalized_operator.A,
            C=penalized_operator.C,
            d=penalized_operator.d,
            small_indices=penalized_operator.small_indices,
            structured_indices=penalized_operator.structured_indices,
            term_name=system.dominant_group_name,
        )
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    centered_data_operator = CenteredBlockOperator(
        raw=data_operator,
        cross=xtw,
        total=system.sum_w,
        center=xtw / system.sum_w,
        raw_structured_cross=(
            system.raw_xtw_structured
            if isinstance(system, SumToZeroBlockStructuredSystem)
            else None
        ),
    )

    return StructuredLinearSystemState(
        coefficient_factor=coefficient_factor,
        profiled_factor=factor,
        augmented_factor=factor.augmented_factor,
        system=system,
        penalized_operator=penalized_operator,
        centered_data_operator=centered_data_operator,
        support_totals=support_totals,
        fallback_reason=getattr(factor, "fallback_reason", None),
    )


def _structured_information_by_group(cache: dict) -> dict[int, np.ndarray]:
    """Reuse dominant Fisher blocks already assembled by a structured refit."""
    system = cache.get("structured_system")
    if isinstance(system, ScalarStructuredSystem):
        return {system.dominant_group_index: system.operator.d}
    if isinstance(
        system,
        BlockStructuredSystem | SumToZeroBlockStructuredSystem,
    ):
        return {system.dominant_group_index: system.operator.D}
    return {}


def _build_reml_reporting_support_state(
    model,
    *,
    result,
    y,
    sample_weight,
    offset_arr,
    durable_retain_fit_state: bool | None = None,
    force: bool = False,
    information_by_group_index: dict[int, np.ndarray] | None = None,
):
    """Distill structured report support under the durable retention contract."""
    retain_reporting_rows = (
        bool(getattr(model, "_retain_fit_state", True))
        if durable_retain_fit_state is None
        else bool(durable_retain_fit_state)
    )
    if retain_reporting_rows and not force:
        return None
    return build_reporting_support_state(
        dm=model._dm,
        groups=model._groups,
        result=result,
        distribution=model._distribution,
        link=model._link,
        sample_weight=sample_weight,
        y=y,
        offset=offset_arr,
        retain_fit_state=retain_reporting_rows,
        information_by_group_index=information_by_group_index,
    )


def restore_qp_group_state(model, qp_saved_state) -> None:
    """Restore monotone-engine/constraint state for QP passthrough groups."""
    for gi, engine, constraints in qp_saved_state:
        model._groups[gi].monotone_engine = engine
        model._groups[gi].constraints = constraints


def compute_profiled_phi(
    model,
    *,
    y,
    sample_weight,
    lambdas,
    reml_penalties,
    pirls_result,
    likelihood_size: float | None = None,
    gamma_scale_data: GammaScaleProfileData | None = None,
) -> float:
    """Return REML-profiled phi for estimated-scale families."""
    scale_known = getattr(model._distribution, "scale_known", True)
    if scale_known:
        return 1.0

    p_dim = model._dm.p
    pq_final = total_penalty_quadratic(
        pirls_result.beta,
        lambdas,
        reml_penalties,
        list(model._dm.group_matrices),
    )
    penalized_deviance = float(pirls_result.deviance + pq_final)

    distribution = model._distribution
    if isinstance(distribution, Gaussian | Gamma):
        hessian_rank = pirls_result.reml_hessian_rank
        if hessian_rank is None:
            hessian_rank = 1 + p_dim
        M_p = compute_penalty_nullity(
            None,
            hessian_rank=hessian_rank,
            penalties=reml_penalties,
            lambdas=lambdas,
            coefficient_width=p_dim,
        )
        if isinstance(distribution, Gaussian):
            if likelihood_size is None:
                likelihood_size = float(np.sum(sample_weight, dtype=np.float64))
            assert likelihood_size is not None
            return profile_gaussian_reml_scale(
                penalized_deviance,
                likelihood_size,
                M_p,
            ).phi
        if gamma_scale_data is None:
            gamma_scale_data = prepare_gamma_reml_scale_data(y, sample_weight)
        return profile_gamma_reml_scale(
            gamma_scale_data,
            penalized_deviance,
            M_p,
        ).phi

    # Reduced profile for estimated-scale families without an explicit Wood
    # Eq. (4) profiler (notably Tweedie). The residual likelihood dimension is
    # governed by the penalty *nullity* on the identified coefficient range,
    # not by the total penalty rank.
    hessian_rank = pirls_result.reml_hessian_rank
    if hessian_rank is None:
        hessian_rank = 1 + p_dim
    M_p = compute_penalty_nullity(
        None,
        hessian_rank=hessian_rank,
        penalties=reml_penalties,
        lambdas=lambdas,
        coefficient_width=p_dim,
    )
    return float(max(penalized_deviance / max(len(y) - M_p, 1.0), 1e-10))


def maybe_qp_passthrough_refit(
    model,
    *,
    qp_passthrough: bool,
    qp_saved_state,
    y,
    sample_weight,
    offset_arr,
    lambdas,
    pirls_result,
    max_pirls_iter,
    pirls_tol,
    reml_penalties,
    trace_run: TraceRun | None = None,
):
    """Run the constrained post-REML refit for QP passthrough flows when needed."""
    if not qp_passthrough:
        return pirls_result

    restore_qp_group_state(model, qp_saved_state)
    qp_output = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        offset=offset_arr,
        beta_init=pirls_result.beta,
        intercept_init=float(pirls_result.intercept),
        max_iter=max_pirls_iter,
        tol=pirls_tol,
        convergence="deviance",
        direct_solve=model._direct_solve,
        reml_penalties=reml_penalties,
        trace_run=trace_run,
        trace_purpose="reml_qp_final",
    )
    return qp_output[0]


def finalize_reml_fit(
    model,
    *,
    best,
    use_direct: bool,
    reml_groups,
    reml_penalties,
    y,
    sample_weight,
    offset,
    offset_arr,
    max_pirls_iter,
    pirls_tol,
    qp_passthrough: bool,
    qp_saved_state,
    profile: dict,
    total_start: float,
    compute_fit_stats,
    trace_run: TraceRun | None = None,
    durable_retain_fit_state: bool | None = None,
):
    """Finalize model state after a successful REML optimization run."""
    model._result = best.pirls_result
    model._reml_lambdas = best.lambdas

    if not use_direct:
        reml_penalties, _, _ = build_penalty_context(model._dm.group_matrices, reml_groups)
    model._reml_penalties = reml_penalties
    model._reml_result = best
    lambdas = best.lambdas
    n_reml_iter = best.n_reml_iter
    converged = best.converged

    solver_result = best.pirls_result
    final_xtwx = None
    final_factor = None
    final_cache: dict = {}
    terminal_curvature = None
    if use_direct:
        terminal_curvature = best.curvature_source
        if terminal_curvature is None:
            terminal_curvature = (
                "fisher"
                if model._discrete
                else classify_reml_curvature(model._distribution, model._link)
            )
        best.curvature_source = terminal_curvature
    if use_direct:
        old_gms = model._dm.group_matrices
        model._dm = rebuild_dm_with_lambdas(model, lambdas, sample_weight)
        reml_penalties, _, _ = build_penalty_context(model._dm.group_matrices, reml_groups)
        model._reml_penalties = reml_penalties

        beta_init = _map_beta_between_bases(
            solver_result.beta,
            old_gms,
            model._dm.group_matrices,
            model._groups,
        )
        observed_terminal = terminal_curvature == "observed" and not qp_passthrough
        final_tolerance = min(pirls_tol, 1e-10) if observed_terminal else pirls_tol
        final_output = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset_arr,
            beta_init=beta_init,
            intercept_init=float(best.pirls_result.intercept),
            max_iter=max_pirls_iter,
            tol=final_tolerance,
            convergence="coefficients" if observed_terminal else "deviance",
            return_xtwx=True,
            cache_out=final_cache,
            direct_solve=model._direct_solve,
            reml_penalties=reml_penalties,
            trace_run=trace_run,
            trace_purpose="reml_final",
        )
        if len(final_output) != 3:  # pragma: no cover - return_xtwx contract
            raise RuntimeError("terminal direct REML refit omitted its working Gram")
        solver_result, final_factor, final_xtwx = final_output

    final_pirls = maybe_qp_passthrough_refit(
        model,
        qp_passthrough=qp_passthrough,
        qp_saved_state=qp_saved_state,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        lambdas=lambdas,
        pirls_result=solver_result,
        max_pirls_iter=max_pirls_iter,
        pirls_tol=pirls_tol,
        reml_penalties=reml_penalties,
        trace_run=trace_run,
    )
    structured_terminal = not qp_passthrough and isinstance(
        final_factor,
        (
            ProfiledScalarSchurFactor,
            ProfiledBlockSchurFactor,
            ProfiledSumToZeroBlockFactor,
        ),
    )
    # Profiled-family publication may retain rows transiently so it can
    # synchronize phi and fit statistics after this refit. Compact reporting
    # must still follow the durable public retention contract.
    reporting_state = _build_reml_reporting_support_state(
        model,
        result=final_pirls,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        durable_retain_fit_state=durable_retain_fit_state,
        force=structured_terminal,
        information_by_group_index=_structured_information_by_group(final_cache),
    )
    structured_linear_state = (
        _build_structured_linear_system_state(
            factor=final_factor,
            data_operator=final_xtwx,
            cache=final_cache,
            support_totals=({} if reporting_state is None else reporting_state.support_totals),
        )
        if use_direct and not qp_passthrough
        else None
    )

    terminal_evaluation: REMLObjectiveEvaluation | None = None
    terminal_tensor_pair_evaluations = None
    if use_direct and model._discrete:
        tensor_pair_summaries = build_tensor_pair_logdet_summaries(
            model._dm.group_matrices,
            reml_penalties,
        )
        terminal_tensor_pair_evaluations = evaluate_tensor_pair_logdet_summaries(
            tensor_pair_summaries,
            lambdas,
        )
    if qp_passthrough:
        # Lambda selection ran on the unconstrained surrogate, but the state
        # published below is the constrained Fisher/QP refit. Its determinant,
        # rank, scale profile, objective, and curvature label must therefore be
        # recomputed as one coherent terminal evaluation. Retaining the
        # optimizer's observed label/objective would describe coefficients that
        # are no longer installed.
        terminal_curvature = "fisher"
        best.curvature_source = terminal_curvature
        if final_pirls.log_det_H is None or final_pirls.reml_hessian_rank is None:
            raise RuntimeError("terminal QP REML refit omitted its Fisher geometry")
        S_final = build_penalty_matrix(
            model._dm.group_matrices,
            model._groups,
            lambdas,
            model._dm.p,
            reml_penalties=reml_penalties,
        )
        terminal_value = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            final_pirls,
            lambdas,
            sample_weight,
            offset_arr,
            log_det_H=final_pirls.log_det_H,
            hessian_rank=final_pirls.reml_hessian_rank,
            S_override=S_final,
            reml_penalties=reml_penalties,
            tensor_pair_evaluations=terminal_tensor_pair_evaluations,
            return_evaluation=True,
        )
        if not isinstance(terminal_value, REMLObjectiveEvaluation):  # pragma: no cover
            raise RuntimeError("terminal QP REML evaluation omitted its scale state")
        terminal_evaluation = terminal_value
        best.objective = terminal_evaluation.value
    if terminal_curvature == "observed" and not qp_passthrough:
        if not final_pirls.converged:
            raise RuntimeError("terminal observed REML refit did not converge")
        S_final = (
            None
            if structured_linear_state is not None
            else build_penalty_matrix(
                model._dm.group_matrices,
                model._groups,
                lambdas,
                model._dm.p,
                reml_penalties=reml_penalties,
            )
        )
        geometry_start = _time.perf_counter()
        terminal_geometry = build_observed_reml_geometry(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset_arr,
            result=final_pirls,
            penalty=S_final,
            derivative_order=0,
            compute_inverse=False,
            groups=model._groups if structured_linear_state is not None else None,
            lambdas=lambdas if structured_linear_state is not None else None,
            reml_penalties=reml_penalties if structured_linear_state is not None else None,
            structured_group_index=(
                structured_linear_state.system.dominant_group_index
                if structured_linear_state is not None
                else None
            ),
        )
        profile["reml_terminal_observed_geometry_s"] = _time.perf_counter() - geometry_start
        mode_score = observed_penalized_mode_score(
            dm=model._dm,
            distribution=model._distribution,
            link=model._link,
            y=y,
            sample_weight=sample_weight,
            result=final_pirls,
            penalty=S_final,
            geometry=terminal_geometry,
            lambdas=lambdas if structured_linear_state is not None else None,
            reml_penalties=reml_penalties if structured_linear_state is not None else None,
        )
        terminal_mode_tolerance = max(
            10.0 * min(pirls_tol, 1e-10),
            100.0 * np.finfo(float).eps,
        )
        profile["reml_terminal_observed_mode_residual"] = mode_score.relative_max
        if mode_score.relative_max > terminal_mode_tolerance:
            raise RuntimeError(
                "terminal observed REML refit does not satisfy the penalized mode "
                f"condition (relative score={mode_score.relative_max:.3e}, "
                f"tolerance={terminal_mode_tolerance:.3e})"
            )
        final_pirls = replace(
            final_pirls,
            log_det_H=terminal_geometry.log_det_H,
            reml_hessian_rank=terminal_geometry.hessian_rank,
        )
        terminal_value = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            final_pirls,
            lambdas,
            sample_weight,
            offset_arr,
            XtWX=final_xtwx,
            log_det_H=terminal_geometry.log_det_H,
            hessian_rank=terminal_geometry.hessian_rank,
            S_override=S_final,
            reml_penalties=reml_penalties,
            tensor_pair_evaluations=terminal_tensor_pair_evaluations,
            return_evaluation=True,
        )
        if not isinstance(terminal_value, REMLObjectiveEvaluation):  # pragma: no cover
            raise RuntimeError("terminal observed REML evaluation omitted its scale state")
        terminal_evaluation = terminal_value
        best.objective = terminal_evaluation.value

    if use_direct and terminal_evaluation is None:
        # The optimizer's retained objective belongs to its retained
        # coefficient state.  The authoritative final refit above can move
        # those coefficients even at unchanged lambdas, so Fisher paths must
        # evaluate LAML once more from the state that will be published.
        if final_xtwx is None:  # pragma: no cover - final direct-refit contract
            raise RuntimeError("terminal Fisher REML refit omitted its working Gram")
        if final_pirls.log_det_H is None or final_pirls.reml_hessian_rank is None:
            raise RuntimeError("terminal Fisher REML refit omitted its retained geometry")
        S_final = (
            None
            if structured_linear_state is not None
            else build_penalty_matrix(
                model._dm.group_matrices,
                model._groups,
                lambdas,
                model._dm.p,
                reml_penalties=reml_penalties,
            )
        )
        terminal_value = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            final_pirls,
            lambdas,
            sample_weight,
            offset_arr,
            XtWX=final_xtwx,
            log_det_H=final_pirls.log_det_H,
            hessian_rank=final_pirls.reml_hessian_rank,
            S_override=S_final,
            reml_penalties=reml_penalties,
            tensor_pair_evaluations=terminal_tensor_pair_evaluations,
            return_evaluation=True,
        )
        if not isinstance(terminal_value, REMLObjectiveEvaluation):  # pragma: no cover
            raise RuntimeError("terminal Fisher REML evaluation omitted its scale state")
        terminal_evaluation = terminal_value
        best.objective = terminal_evaluation.value

    # Profile dispersion from the state that will actually be returned.  A
    # constrained QP passthrough refit can change both beta'S beta and deviance.
    if isinstance(model._distribution, Tweedie) and not qp_passthrough:
        phi_fixed = float(final_pirls.phi)
    elif terminal_evaluation is not None and terminal_evaluation.profiled_scale is not None:
        phi_fixed = terminal_evaluation.profiled_scale.phi
    elif terminal_evaluation is not None and not getattr(
        model._distribution,
        "scale_known",
        True,
    ):
        penalty_nullity = float(terminal_evaluation.penalty_nullity or 0.0)
        phi_fixed = max(
            float(terminal_evaluation.penalized_deviance)
            / max(float(len(y)) - penalty_nullity, 1.0),
            1.0e-10,
        )
    else:
        phi_fixed = compute_profiled_phi(
            model,
            y=y,
            sample_weight=sample_weight,
            lambdas=lambdas,
            reml_penalties=reml_penalties,
            pirls_result=final_pirls,
        )

    corrected = replace(final_pirls, phi=phi_fixed)
    model._result = corrected
    model._reml_result.pirls_result = corrected
    model._reporting_support_state = reporting_state
    model._linear_system_state = structured_linear_state

    update_reml_r_inv(model, reml_groups, lambdas)

    profile["total_s"] = _time.perf_counter() - total_start
    profile["n_reml_iter"] = n_reml_iter
    profile["converged"] = converged
    model._reml_profile = profile

    eta = model._dm.matvec(model._result.beta) + model._result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = compute_fit_stats(
        y, mu, sample_weight, offset, model._distribution, model._link, model._result.phi
    )
    model._solver_result = corrected

    meta = {"method": "fit_reml", "discrete": model._discrete}
    meta["direct_backend"] = corrected.direct_backend
    meta["direct_fallback_reason"] = corrected.direct_fallback_reason
    if qp_passthrough:
        meta["lambda_strategy"] = "qp_passthrough"
    model._last_fit_meta = meta

    restore_qp_group_state(model, qp_saved_state)
    return lambdas, n_reml_iter, converged
