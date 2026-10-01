"""Internal REML fit finalization helpers."""

from __future__ import annotations

import time as _time
import warnings
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
from superglm.reml.identified import (
    IdentifiedLaplace,
    WeakIdentificationWarning,
    coefficient_labels,
    dense_hessian,
    final_mode_weak_slopes,
)
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.observed_geometry import (
    ObservedGeometryInfeasibleError,
    ObservedModeNotConvergedError,
    build_observed_reml_geometry,
    classify_reml_curvature,
    mode_certification_hint,
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
from superglm.solvers.mode_score import linear_predictor, mode_certification_bar
from superglm.solvers.structured import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    FactorSmoothLeafSystem,
    NestedDataOperator,
    NestedPenalizedOperator,
    NestedStructuredSystem,
    ProfiledFactorSmoothLeafFactor,
    ProfiledNestedSchurFactor,
    ProfiledSumToZeroTreeFactor,
    StructuredLinearSystemState,
    SumToZeroBlockOperator,
    SumToZeroLeafSystem,
    centred_data_operator,
    release_leaf_memo,
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
        ProfiledFactorSmoothLeafFactor | ProfiledSumToZeroTreeFactor | ProfiledNestedSchurFactor,
    ):
        return None
    system = cache.get("structured_system")
    penalized_operator = cache.get("penalized_operator")
    if not isinstance(
        system,
        FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem,
    ) or not isinstance(
        penalized_operator,
        BlockSymmetricOperator | SumToZeroBlockOperator | NestedPenalizedOperator,
    ):
        raise RuntimeError("terminal structured refit omitted its compact system state")
    if not isinstance(
        data_operator,
        BlockSymmetricOperator | SumToZeroBlockOperator | NestedDataOperator,
    ):
        raise RuntimeError("terminal structured refit omitted its compact data operator")

    # No raw-coordinate coefficient factor and no view derived from one (design
    # §3.6, §3.10): the slope covariance is M_ss, the slope block of the
    # augmented inverse, which ``profiled_factor`` serves.
    if isinstance(system, FactorSmoothLeafSystem | SumToZeroLeafSystem):
        if data_operator is not system.operator:
            raise RuntimeError("terminal structured refit's data operator is not its system's")
        # on the c0-shifted moments (design §3.2): the estimability, the column
        # scales and the small data factor read its numbers
        centered_data_operator = centred_data_operator(
            system, row_column_norm=cache.get("structured_row_column_norm")
        )
        # the published state keeps no row-scale leaf arrays (Opus review P1):
        # the system and the factor are copies whose leaf keeps only what a
        # published factor reads, and the fit's own objects are left as they are
        own = factor.augmented_factor.system is system
        system = system.published()
        factor = factor.published(system if own else None)
    else:
        xtw = np.empty(system.operator.shape[0], dtype=np.float64)
        xtw[system.operator.small_indices] = system.xtw_small
        xtw[system.operator.structured_indices] = system.xtw_structured
        centered_data_operator = CenteredBlockOperator(
            raw=data_operator,
            cross=xtw,
            total=system.sum_w,
            center=xtw / system.sum_w,
            row_column_norm=cache.get("structured_row_column_norm"),
        )

    return StructuredLinearSystemState(
        profiled_factor=factor,
        augmented_factor=factor.augmented_factor,
        system=system,
        penalized_operator=penalized_operator,
        centered_data_operator=centered_data_operator,
        support_totals=support_totals,
    )


def _structured_information_by_group(cache: dict) -> dict[int, np.ndarray]:
    """Reuse dominant Fisher blocks already assembled by a structured refit.

    Every nested chain level reports its node weights, the subtree sums of
    the leaf information.
    """
    system = cache.get("structured_system")
    if isinstance(system, NestedStructuredSystem):
        return dict(
            zip(
                system.chain_group_indices,
                system.operator.tree.split(system.xtw_structured),
                strict=True,
            )
        )
    if isinstance(
        system,
        FactorSmoothLeafSystem | SumToZeroLeafSystem,
    ):
        return {system.dominant_group_index: system.operator.D}
    return {}


def _structured_geometry_groups(
    state: StructuredLinearSystemState | None,
) -> tuple[int | None, tuple[int, ...]]:
    """Return the leaf group index and the nested chain (``()`` for one level)."""
    if state is None:
        return None, ()
    if isinstance(state.system, NestedStructuredSystem):
        return state.system.chain_group_indices[-1], state.system.chain_group_indices
    return state.system.dominant_group_index, ()


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
    direct_solve: str,
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
        direct_solve=direct_solve,
        reml_penalties=reml_penalties,
        trace_run=trace_run,
        trace_purpose="reml_qp_final",
        weight_semantics=model_weight_semantics(model),
    )
    return qp_output[0]


def _disclose_thin_levels(profile: dict, factor) -> None:
    """Name the ``sz`` levels too thin to identify their polynomial deviation (decision 4).

    A level with fewer than ``m`` distinct weighted ``x`` values (the penalty's
    null space) beside the required global Spline makes an exact alias of the
    main effect's polynomial with that level's deviation, and a level with no
    weight lets the others' constants reproduce the intercept.  The balance
    tree keeps the level and truncates the alias (one-engine design §3.5):
    ``profile["structured_thin_levels"]`` and one ``UserWarning`` name it.
    """
    levels = tuple(getattr(factor, "thin_levels", ()) or ())
    if not levels:
        return
    profile["structured_thin_levels"] = levels
    names = ", ".join(str(level) for level in levels)
    warnings.warn(
        f"FactorSmooth term {factor.dominant_group_name!r} (basis='sz') has levels with fewer "
        f"distinct x values than its penalty's null space, or no weight: {names}. Each such "
        "level's deviation is aliased with the main effect along the polynomial the penalty "
        "does not shrink. The fit keeps the level; the coefficients the alias touches, which "
        "can be every coefficient of this term and of its main effect, have no standard error "
        "(NaN).",
        UserWarning,
        stacklevel=4,
    )


def _disclose_weak_identification(
    model,
    *,
    profile: dict,
    identified: IdentifiedLaplace,
    result,
    factor,
    sample_weight,
    offset_arr,
    lambdas,
    reml_penalties,
    at_mode: bool,
) -> None:
    """Publish every weakly identified slope of a finished REML fit (design §3.9).

    Once per fit, for every family and backend of the direct REML engine: the
    §3.9 test at the final mode from its Fisher rows
    (``reml.identified.final_mode_weak_slopes``; ``at_mode``, not for a
    constrained QP refit, whose mode its QP certifies), the slopes the
    Laplace approximation left out, the directions the border factor
    truncated as weakly identified and those the observed certificate
    flagged.  The fit keeps them all; ``profile["reml_weakly_identified"]``
    holds the slope indices, ``model.diagnostics()`` their names, and a
    ``WeakIdentificationWarning`` names them once.
    """
    flagged: set[int] = set()
    if at_mode:
        flagged.update(
            int(index)
            for index in final_mode_weak_slopes(
                dm=model._dm,
                distribution=model._distribution,
                link=model._link,
                sample_weight=sample_weight,
                offset_arr=offset_arr,
                result=result,
                lambdas=lambdas,
                penalties=reml_penalties,
            )
        )
    flagged.update(int(index) for index in identified.excluded)
    # A factor that tells the columns a weak direction names from those it
    # moves (an sz balance tree) discloses the named ones; the certificate
    # left every moved column ungated, so its own flags count beyond those.
    moved = {int(index) for index in getattr(factor, "weakly_identified_slopes", ()) or ()}
    named = getattr(factor, "weakly_identified_named_slopes", None)
    flagged.update(moved if named is None else (int(index) for index in named))
    _disclose_thin_levels(profile, factor)
    flagged.update(
        int(index)
        for index in profile.get("reml_terminal_weakly_identified", ()) or ()
        if int(index) not in moved
    )
    indices = tuple(sorted(flagged))
    labels = coefficient_labels(model._groups, indices)
    profile["reml_weakly_identified"] = indices
    profile["reml_weakly_identified_labels"] = labels
    profile["reml_laplace_excluded"] = tuple(int(index) for index in identified.excluded)
    excluded_labels = coefficient_labels(
        model._groups, tuple(int(index) for index in identified.excluded)
    )
    profile["reml_laplace_excluded_labels"] = excluded_labels
    if labels:
        left_out = (
            f" Left out of smoothing-parameter selection: {', '.join(excluded_labels)}."
            if excluded_labels
            else " None of them is left out of smoothing-parameter selection."
        )
        warnings.warn(
            "These coefficients carry information only at the noise level of the data "
            "(a factor level or column with little or no weight, observations or "
            f"information): {', '.join(labels)}. They stay in the model, and their "
            f"estimates and standard errors carry little information.{left_out}",
            WeakIdentificationWarning,
            stacklevel=4,
        )


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
    # The terminal refit takes the backend the optimizer did: one decision
    # from the model's terms (``resolve_structured_backend``), never switched.
    direct_solve = model._direct_solve
    final_xtwx = None
    final_factor = None
    final_cache: dict = {}
    # Whether the terminal refit met the mode certificate (None: a route the
    # certificate does not judge -- constrained, SCOP, QP passthrough).
    terminal_certified: bool | None = None
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
    # The Laplace approximation's identified part (design §3.9,
    # ``reml.identified``): the terminal objective leaves out the same slopes
    # the optimizer's did, so it scores the state it publishes consistently.
    identified = IdentifiedLaplace()
    if use_direct:
        old_gms = model._dm.group_matrices
        model._dm = rebuild_dm_with_lambdas(model, lambdas, sample_weight)
        reml_penalties, _, _ = build_penalty_context(
            model._dm.group_matrices,
            reml_groups,
            _reuse_raw_from=reml_penalties,
            _reuse_fixed_from=reml_penalties,
        )
        model._reml_penalties = reml_penalties
        if getattr(best, "reml_penalties", None) is not None:
            # The carrier must not keep obsolete optimizer owners and their
            # weighted evaluations alive alongside the terminal context.
            best.reml_penalties = reml_penalties

        beta_init = _map_beta_between_bases(
            solver_result.beta,
            old_gms,
            model._dm.group_matrices,
            model._groups,
        )
        if not qp_passthrough:
            identified = IdentifiedLaplace.for_design(model._dm, sample_weight, reml_penalties)
        observed_terminal = terminal_curvature == "observed" and not qp_passthrough
        # One-engine design §3.8: the terminal refit of every route auto uses,
        # exact and discrete, gram and structured, stops on the certificate's
        # own centred score over the identified coefficients, at the one bar
        # (``mode_score.MODE_CERTIFICATION_BAR``).  An objective change is
        # blind to a coefficient whose rows carry a small share of the
        # likelihood (a level driven towards its penalty bound, a tiny-lambda
        # direction), whose log-curvature still enters log|H| at full weight,
        # so the published objective described a point short of the mode
        # (stage-0 verifier census: raw_1e8 binomial, 21.3 above it; the
        # discrete terminal's unconstrained score left raw_1e8 binomial 16
        # above gram's, stage-1 verifier).  The bar is set by what the REML
        # criterion needs, so Fisher's linear rate on a non-canonical link
        # reaches it; a refit that does not is published as not converged,
        # never refused.  A linearly constrained or SCOP mode is certified by
        # its inner QP/KKT state and keeps the objective stop.
        shaped = any(
            getattr(group, "constraints", None) is not None
            or getattr(group, "monotone_engine", None) == "scop"
            for group in model._groups
        )
        certified_terminal = observed_terminal or (not qp_passthrough and not shaped)
        final_tolerance = min(pirls_tol, 1e-10) if certified_terminal else pirls_tol
        terminal_bar = mode_certification_bar(profile.get("reml_tol_resolved"))
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
            convergence="mode_score" if certified_terminal else "deviance",
            return_xtwx=True,
            cache_out=final_cache,
            direct_solve=direct_solve,
            reml_penalties=reml_penalties,
            trace_run=trace_run,
            trace_purpose="reml_final",
            weight_semantics=model_weight_semantics(model),
            _laplace_excluded=tuple(int(index) for index in identified.excluded),
            _mode_bar=terminal_bar,
        )
        if len(final_output) != 3:  # pragma: no cover - return_xtwx contract
            raise RuntimeError("terminal direct REML refit omitted its working Gram")
        solver_result, final_factor, final_xtwx = final_output
        if certified_terminal:
            terminal_certified = bool(solver_result.converged)

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
        direct_solve=direct_solve,
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
            ProfiledFactorSmoothLeafFactor,
            ProfiledSumToZeroTreeFactor,
            ProfiledNestedSchurFactor,
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
            # penalized mode. Only a structural failure (a non-finite
            # deviance, an infeasible or KKT-incomplete constrained mode)
            # lands here; a mode short of the certificate is published as
            # not converged (``terminal_certified``), as at the candidate gate.
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
        structured_group_index, structured_chain = _structured_geometry_groups(
            structured_linear_state
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
                # the identified part rebuilds the structured factor returned here
                compute_inverse=bool(identified),
                groups=model._groups if structured_linear_state is not None else None,
                lambdas=lambdas if structured_linear_state is not None else None,
                reml_penalties=reml_penalties if structured_linear_state is not None else None,
                structured_group_index=structured_group_index,
                structured_chain_group_indices=structured_chain,
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
        objective_geometry = identified.geometry(terminal_geometry)
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
                excluded=identified.excluded,
                bar=terminal_bar,
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
        # The certificate is the terminal PIRLS's own stop residual
        # (``final_pirls.converged`` under ``convergence="mode_score"``); the
        # geometry's score here is published as a diagnostic, never gated.
        profile["reml_terminal_observed_mode_residual"] = mode_score.relative_max
        # §3.8, §3.9: disclose a binding floor and every coefficient flagged
        # weakly identified and kept (slope indices), which the certificate
        # did not gate.
        profile["reml_terminal_mode_floor_binding"] = mode_score.floor_binding
        profile["reml_terminal_weakly_identified"] = mode_score.weakly_identified
        profile["reml_terminal_raw_mode_residual"] = mode_score.raw_relative_max
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
            log_det_H=objective_geometry.log_det_H,
            hessian_rank=objective_geometry.hessian_rank,
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
            log_det_H=identified.log_det(
                final_factor, final_pirls.log_det_H, dense_hessian(final_cache)
            ),
            hessian_rank=identified.rank(
                final_pirls.reml_hessian_rank, final_factor, dense_hessian(final_cache)
            ),
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
    _disclose_weak_identification(
        model,
        profile=profile,
        identified=identified,
        result=corrected,
        factor=final_factor,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        lambdas=lambdas,
        reml_penalties=reml_penalties,
        at_mode=use_direct and not qp_passthrough,
    )
    model._reporting_support_state = reporting_state
    model._linear_system_state = structured_linear_state
    release_leaf_memo(getattr(model._dm, "_structured_layout_cache", None))

    update_reml_r_inv(model, reml_groups, lambdas)

    if terminal_certified is not None:
        # Owner decision 3 (2026-09-30): a terminal mode that stays short of
        # the certificate is published as not converged, never refused.
        profile["reml_terminal_mode_certified"] = terminal_certified
        profile["reml_terminal_mode_termination"] = solver_result.termination_reason
        if not terminal_certified:
            converged = False
            best.converged = False
    profile["total_s"] = _time.perf_counter() - total_start
    profile["n_reml_iter"] = n_reml_iter
    profile["converged"] = converged
    model._reml_profile = profile

    eta = linear_predictor(model._dm, model._result, offset)
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution, model._link)

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
