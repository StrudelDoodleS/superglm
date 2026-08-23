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
from superglm.model.reml_setup import restore_qp_constraints
from superglm.model.reml_state import update_reml_r_inv
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.observed_geometry import (
    ObservedGeometryInfeasibleError,
    ObservedModeNotCertifiedError,
    ObservedModeNotConvergedError,
    build_observed_reml_geometry,
    classify_reml_curvature,
    mode_certification_hint,
    observed_mode_certification_bar,
    observed_penalized_mode_score,
    stopped_on_iteration_budget,
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
    gaussian_reml_scale_terms,
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
)
from superglm.solvers.dispersion import dispersion_likelihood_size, model_weight_semantics
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
    if bool(getattr(model, "_suppress_reporting_support", False)):
        return None
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
    restore_qp_constraints(model, qp_saved_state)


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
        weight_semantics = model_weight_semantics(model)
        if isinstance(distribution, Gaussian):
            saturated_log_weight = 0.0
            if likelihood_size is None:
                likelihood_size, saturated_log_weight = gaussian_reml_scale_terms(
                    sample_weight,
                    weight_semantics=weight_semantics,
                )
            assert likelihood_size is not None
            return profile_gaussian_reml_scale(
                penalized_deviance,
                likelihood_size,
                M_p,
                saturated_log_weight=saturated_log_weight,
            ).phi
        if gamma_scale_data is None:
            gamma_scale_data = prepare_gamma_reml_scale_data(
                y,
                sample_weight,
                weight_semantics=weight_semantics,
            )
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
    # The declared contract's likelihood size, not the row count. This
    # fallback is reached by `apply_shape_postfit`'s repair as well as the
    # terminal publication, so a row-count denominator here republishes the
    # wrong dispersion -- and every Wald interval drawn from it -- after an
    # otherwise contract-correct fit.
    size = (
        float(likelihood_size)
        if likelihood_size is not None
        else dispersion_likelihood_size(
            sample_weight,
            weight_semantics=model_weight_semantics(model),
        )
    )
    return float(max(penalized_deviance / max(size - M_p, 1.0), 1e-10))


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
        weight_semantics=model_weight_semantics(model),
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
    # The optimizer already built and filled a per-fit Tweedie saturated-density
    # cache; without this the terminal evaluations below construct a cold one and
    # re-solve a (Dp, Mp) the search has already solved (measured: 20 of 177
    # fresh density passes on a burn-cost-shaped fit).  Same pure inputs, so the
    # reconstructed object is value-identical -- only its memo dicts differ.
    terminal_tweedie_scale_data = getattr(best, "tweedie_scale_data", None)
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
            weight_semantics=model_weight_semantics(model),
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
    # Typed for the same routing contract as the candidate-side gate in
    # run_fixed_monotone_reml: a power search treats a terminal QP refit
    # with no feasible mode as this point's infeasibility, not a crash.
    if final_pirls.termination_reason == "constraint_infeasible":
        raise ObservedModeNotConvergedError(
            "terminal constrained REML refit ended at an infeasible coefficient mode"
        )
    if final_pirls.termination_reason == "constraint_kkt_incomplete":
        raise ObservedModeNotConvergedError(
            "terminal constrained REML refit ended without a complete inner-QP KKT certificate"
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
            tweedie_scale_data=terminal_tweedie_scale_data,
            weight_semantics=model_weight_semantics(model),
            return_evaluation=True,
        )
        if not isinstance(terminal_value, REMLObjectiveEvaluation):  # pragma: no cover
            raise RuntimeError("terminal QP REML evaluation omitted its scale state")
        terminal_evaluation = terminal_value
        best.objective = terminal_evaluation.value
    if terminal_curvature == "observed" and not qp_passthrough:
        if not final_pirls.converged and not stopped_on_iteration_budget(final_pirls):
            # Typed: to a power search this is one more point with no usable
            # penalized mode. The terminal refit runs at the FINAL lambda,
            # which differs from every trial lambda, so this door is reachable
            # even for a point whose candidate fits all certified. It warm
            # starts at the same 1e-10 step bar, so it is exposed to the same
            # round-off floor as the candidate gate and defers to the same
            # certificate below. No draw reaching it has been produced; the
            # two gates are kept identical rather than left to diverge.
            raise ObservedModeNotConvergedError(
                "terminal observed REML refit did not converge to a penalized coefficient mode",
                hint=mode_certification_hint(model._distribution),
            )
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
        try:
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
        except ObservedGeometryInfeasibleError as exc:
            # The same retype the candidate gate in optimize_direct_reml makes,
            # for the same reason. The convergence gate above now admits a
            # budget-exhausted iterate, and an iterate still mid-descent can
            # carry observed information the geometry refuses outright -- a
            # non-finite row, a non-positive intercept curvature. The line
            # search answers that refusal by halving its step; here there is no
            # step left to halve, so retyping is the only routing available.
            # Left bare this is a ValueError, and it would sail past every
            # `except ObservedModeNotCertifiedError` that guards this call --
            # the power search in profiling/tweedie.py, which names the
            # terminal refit as a place it expects this family from, and the
            # two publication handlers in model/profile_ops.py -- killing a
            # search that only needed to score this point infeasible.
            #
            # Only that one refusal is retyped. The build's other ValueErrors
            # report a violated caller contract -- a misshapen design, a bad
            # derivative_order, a penalty that is not PSD -- which no iterate
            # can clear, so they must keep surfacing as the bugs they are.
            raise ObservedModeNotConvergedError(
                f"terminal observed REML geometry refused the penalized coefficient mode: {exc}",
                hint=mode_certification_hint(model._distribution),
                infeasible_detail="terminal observed geometry refused the penalized mode",
            ) from exc
        profile["reml_terminal_observed_geometry_s"] = _time.perf_counter() - geometry_start
        try:
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
        except ObservedGeometryInfeasibleError as exc:
            # The score carries the same exposure as the build above: a score
            # that evaluates non-finite describes these coefficients, so it has
            # to reach a power search as one more point with no usable
            # penalized mode rather than as a bare ValueError nothing on that
            # path catches. Its argument-shape refusals stay bare, for the
            # reason the build's do.
            raise ObservedModeNotConvergedError(
                "terminal observed REML refit could not score its penalized "
                f"coefficient mode: {exc}",
                hint=mode_certification_hint(model._distribution),
                infeasible_detail="terminal observed mode score refused the penalized mode",
            ) from exc
        # The same fixed bar the candidate gate uses: a point that certified
        # during the search cannot fail publication because the caller
        # tightened pirls_tol below the observed-geometry ceiling.
        terminal_mode_tolerance = observed_mode_certification_bar()
        profile["reml_terminal_observed_mode_residual"] = mode_score.relative_max
        if mode_score.relative_max > terminal_mode_tolerance:
            # Typed for the same reason as the candidate-fit gate in
            # optimize_direct_reml: a power search must be able to score this
            # point infeasible instead of dying on it.
            raise ObservedModeNotCertifiedError(
                mode_score.relative_max,
                terminal_mode_tolerance,
                hint=mode_certification_hint(model._distribution),
            )
        final_pirls = replace(
            final_pirls,
            log_det_H=terminal_geometry.log_det_H,
            reml_hessian_rank=terminal_geometry.hessian_rank,
        )
        if not final_pirls.converged:
            # Reachable only because the certificate above declined to raise:
            # this mode's KKT residual cleared observed_mode_certification_bar(),
            # a fixed constant no caller tolerance can move. Relabelling it
            # converged strengthens the published claim over the step-length
            # verdict the flag normally carries, so every reader of the
            # published result now agrees with the certificate that admitted
            # the fit. The distinct reason is load-bearing -- nothing may read
            # this as the step test having fired -- and the guard keeps that
            # name off a refit that did converge by step length. fit_state.py
            # rewrites the same pair under its own synthetic reason when a
            # coefficient revision invalidates a mode; the two copies published
            # below are this same object, so public and solver stay consistent.
            final_pirls = replace(
                final_pirls,
                converged=True,
                termination_reason="mode_certified",
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
            tweedie_scale_data=terminal_tweedie_scale_data,
            weight_semantics=model_weight_semantics(model),
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
            tweedie_scale_data=terminal_tweedie_scale_data,
            weight_semantics=model_weight_semantics(model),
            return_evaluation=True,
        )
        if not isinstance(terminal_value, REMLObjectiveEvaluation):  # pragma: no cover
            raise RuntimeError("terminal Fisher REML evaluation omitted its scale state")
        terminal_evaluation = terminal_value
        best.objective = terminal_evaluation.value

    # Profile dispersion from the state that will actually be returned.  A
    # constrained QP passthrough refit can change both beta'S beta and deviance.
    #
    # The deviance-form branches below divide by the DECLARED contract's
    # likelihood size, not the physical row count: `sum(w)` under
    # `"frequency"`, the positive-row count under `"prior"`.  Reading `len(y)`
    # there puts the published dispersion -- and every Wald interval drawn from
    # it -- out of step with both the terminal objective and literal row
    # replication whenever the weights are not all one.
    published_likelihood_size = dispersion_likelihood_size(
        sample_weight,
        weight_semantics=model_weight_semantics(model),
    )
    if isinstance(model._distribution, Tweedie) and not qp_passthrough:
        phi_fixed = float(final_pirls.phi)
    elif isinstance(model._distribution, Tweedie) and terminal_evaluation is not None:
        # QP passthrough keeps publishing the deviance-form dispersion with
        # the terminal evaluation's identified nullity. The evaluation now
        # carries the exact Tweedie scale profile for the criterion itself,
        # but unifying the three published Tweedie dispersions (Pearson here
        # above, profile MLE in estimate_p, deviance form on this path) is
        # deliberately deferred to a release that measures its own
        # before/after; see the families guide's dispersion inventory.
        penalty_nullity = float(terminal_evaluation.penalty_nullity or 0.0)
        phi_fixed = max(
            float(terminal_evaluation.penalized_deviance)
            / max(published_likelihood_size - penalty_nullity, 1.0),
            1.0e-10,
        )
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
            / max(published_likelihood_size - penalty_nullity, 1.0),
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
        y,
        mu,
        sample_weight,
        offset,
        model._distribution,
        model._link,
        model._result.phi,
        weight_semantics=model_weight_semantics(model),
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
