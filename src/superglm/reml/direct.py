"""Direct REML Newton optimizer (exact path).

Damped Newton optimization of the REML/LAML objective with
W(rho) correction, gradient, Hessian, and Armijo line search.

References
----------
- Wood (2011) Section 6.2.
"""

from __future__ import annotations

import math
import time as _time
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._fit_trace import TraceRun
from superglm._lambda_box import LAMBDA_CEILING, LAMBDA_FLOOR
from superglm._roundoff import NEWTON_EIGENVALUE_FLOOR
from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
from superglm.group_matrix import DesignMatrix
from superglm.links import CloglogLink, LogitLink, LogLink, ProbitLink
from superglm.reml._leverage_rows import reads_row_supports
from superglm.reml.convergence import (
    FLAT_DIRECTION_FREEZE_FLOOR,
    classify_dead_feasible_exit,
    concave_live_directions,
    direction_penalty_ranks,
    evaluate_reml_candidate,
    fixed_freeze_record,
    freeze_flat_directions,
    freeze_record,
    mask_frozen_stop_gradient,
    newton_predicted_decrease,
    newton_step_left,
    newton_step_resolved,
    newton_step_settled,
    newton_step_stop_bar,
    project_reml_gradient,
    revalidate_stop,
    trial_counts_as_precision_evidence,
)
from superglm.reml.discrete import bootstrap_seed_lambdas, optimize_discrete_reml_cached_w
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.identified import IdentifiedLaplace, dense_hessian
from superglm.reml.objective import (
    REMLObjectiveEvaluation,
    reml_laml_objective,
)
from superglm.reml.observed_geometry import (
    OBSERVED_PIRLS_TOL_CEILING,
    ObservedGeometryInfeasibleError,
    ObservedModeNotConvergedError,
    ObservedREMLGeometry,
    build_observed_reml_geometry,
    classify_reml_curvature,
    mode_certification_hint,
    observed_penalized_mode_score,
    stopped_on_iteration_budget,
    validate_observed_derivative_capability,
)
from superglm.reml.penalty_algebra import (
    build_penalty_matrix,
    coerce_reml_penalties,
    compute_penalty_nullity,
    penalty_component_quadratic,
    total_penalty_quadratic,
)
from superglm.reml.penalty_support import PenaltyNumericalError
from superglm.reml.result import REMLResult
from superglm.reml.scale import (
    prepare_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
    profile_tweedie_reml_scale,
)
from superglm.reml.w_derivatives import reml_w_correction, validate_w_correction_order
from superglm.solvers._structured.selection import (
    record_auto_backend_decision,
    resolve_structured_backend,
    select_structured_group,
)
from superglm.solvers.centered_system import (
    TabmatCenteringState,
    _FisherDataReuse,
    _InitialDataReuse,
)
from superglm.solvers.dispersion import lazy_likelihood_size
from superglm.solvers.hessian_factor import DenseHessianFactor, as_hessian_factor
from superglm.solvers.irls_direct import StructuredSolverError, fit_irls_direct
from superglm.solvers.mode_score import centred_warm_start, mode_certification_bar
from superglm.types import GroupSlice, PenaltyComponent

# The solve ceiling lives in observed_geometry, shared with the terminal
# publication gate in model/reml_finalize.py.
_OBSERVED_PIRLS_TOL_CEILING = OBSERVED_PIRLS_TOL_CEILING

# One leverage pass replaces one derivative Gram per penalty; below four
# penalties the Gram route is cheaper.
_LEVERAGE_MIN_PENALTIES = 4
# The pre-solve hands the exact loop a start, not a result: it stops once a
# working-model update moves the objective by less than this fraction of its
# scale.
_PRESOLVE_OBJECTIVE_TOL = 1e-7
# The pre-solve's tolerance is never looser than this (a tighter caller
# tolerance passes through): its freeze and stop bars scale with it.
_PRESOLVE_REML_TOL = 1e-9
# The automatic pre-solve runs on a cold exact fit with at least this many
# estimated smoothing parameters, as each exact iteration's weight-correction
# pass grows with that count.
_PRESOLVE_MIN_ESTIMATED = 8
# Nor under a shorter iteration cap: the pre-solve takes up to
# ``max_reml_iter`` updates itself, which would double a capped budget.
_PRESOLVE_MIN_REML_ITER = 10
# Only for these exact family/link types, with no RandomEffect or FactorSmooth
# term the structured solvers could take: on them the cached-W working model
# carries the exact loop's curvature (Fisher on a canonical link, observed rows
# where ``concave_observed_rows`` declares them) and the pre-solve pays.
# Tweedie power-search candidates and estimated-theta NB2 fits start cold
# (``model.reml_ops``).
_PRESOLVE_PAIRS = frozenset(
    {
        (Poisson, LogLink),
        (Binomial, LogitLink),
        (Binomial, ProbitLink),
        (Binomial, CloglogLink),
        (Gamma, LogLink),
        (Tweedie, LogLink),
        (NegativeBinomial, LogLink),
    }
)
# First-order fits keep the derivative Grams (``m n p^2`` multiply-adds per
# iteration) until they pass this work, then take the one-pass leverage route
# (``n p^2``, ``reml_w_correction(leverage_hessian=True)``); Wood (2011, sec.
# 3.5.1) prices the two schemes the same way. The bar reads the dense cost on
# either backend, so a structured-capable model decides alike on both, and
# above it both leave the W(rho) cross terms out (``gradient_terms_only``).
# Binomial/logit on a dense factor takes the leverage route at any work.
# A structured-capable model takes it only where its solved-row pass is sparse
# too (``reads_row_supports``); every other model is priced as dense.
_LEVERAGE_HESSIAN_MIN_GRAM_WORK = 1.0e11


def _usable_w_correction(correction, args: tuple, kwargs: dict, distribution):
    """``correction(*args, **kwargs)``; a ``LinAlgError`` out of it is no usable curvature here."""
    try:
        return correction(*args, **kwargs)
    except np.linalg.LinAlgError as error:
        raise ObservedModeNotConvergedError(
            f"REML W(rho) correction has no usable curvature at this point: {error}",
            hint=mode_certification_hint(distribution),
            # PIRLS reached a mode; the wording must not claim it did not.
            infeasible_detail="the W(rho) correction has no usable curvature",
        ) from error


def _slow_reversal(
    delta: NDArray,
    previous_delta: NDArray | None,
    contraction: float,
    excess_decrease: float,
    obj_change: float,
    obj: float,
    tol: float,
) -> bool:
    """Whether this order-1 Newton step reversed a previous one at a slow, linear rate.

    All of these hold: a coordinate changed sign against the previous step; the
    gradient shrank by less than half a step, but shrank; the objective fell by
    no more than twice what the quadratic model predicted (a larger fall is a
    moving mode, as a separated level drifting to its bound, not curvature);
    and the objective has settled to ``sqrt(tol)``, where a Newton step would
    square the error.
    """
    return (
        previous_delta is not None
        and bool(np.any(delta * previous_delta < 0.0))
        and 0.5 < contraction < 1.0
        and excess_decrease <= 0.0
        and obj_change <= math.sqrt(tol) * (1.0 + abs(obj))
    )


def _escalates_to_exact_hessian(
    step_reversals: int,
    slow_reversal: bool,
    tol: float,
    score_scale: float,
    proj_grad_norm: float,
    contraction: float,
    iterations_left: int,
) -> bool:
    """Whether two slow reversals in a row, at a rate that cannot finish in time, switch Hessians.

    A linear rate that reaches the score bar within the iterations left keeps
    the order-1 Hessian.
    """
    if step_reversals < 2 or not slow_reversal:
        return False
    steps = math.log(tol * score_scale / max(proj_grad_norm, 1e-300)) / math.log(contraction)
    return steps > iterations_left


def _backtracked_rate_cannot_finish(
    gradient_norm: float, score_bar: float, step: float, iterations_left: int
) -> bool:
    """Whether Newton steps taken at fraction ``step`` cannot reach the score bar in time.

    A step at ``step`` leaves about ``1 - step`` of the gradient, so the bar
    is ``ln(bar / |g|) / ln(1 - step)`` iterations away.
    """
    if gradient_norm <= score_bar or not 0.0 < step < 1.0:
        return False
    return math.log(score_bar / gradient_norm) / math.log(1.0 - step) > iterations_left


def _mean_space_boundary_message(rows: int, *, estimated: bool, unresolved: bool = False) -> str:
    """Why a fit whose maximum stays at the mean-space boundary has no REML criterion.

    ``unresolved``: the rows rise towards the boundary along a direction the
    factorization truncates (``TruncatedDirection.boundary``) rather than
    being fitted at it.
    """
    where = (
        "even with every estimated smoothing parameter at its upper bound"
        if estimated
        else "at the fixed smoothing parameters"
    )
    fitted = (
        "rise towards probability 1 along a direction float64 cannot resolve"
        if unresolved
        else "are fitted at probability 1"
    )
    return (
        f"The binomial/log model has no interior maximum: {rows} row(s) {fitted}, the "
        f"boundary of its parameter space, {where}. The maximum "
        "is constrained there rather than stationary, so the Laplace-approximate REML "
        "criterion is not defined at it. The unpenalized part of the model fits these "
        "rows at probability one (for example a categorical level or parametric term "
        "whose rows all have y = 1); a logit link, merging that level, or a Poisson/log "
        "risk-ratio fit with robust standard errors (Zou 2004) avoids the boundary."
    )


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
    weight_semantics: str,
    max_reml_iter: int,
    reml_tol: float,
    verbose: bool,
    penalty_caches: dict | None = None,
    profile: dict | None = None,
    select_snap: bool = True,
    direct_solve: str = "auto",
    w_correction_order: int = 1,
    reml_penalties: list[PenaltyComponent] | None = None,
    estimated_names: set[str] | None = None,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
    debug_recorder=None,
    trace_run: TraceRun | None = None,
    warm_lambdas: Mapping[str, float] | None = None,
    presolve: bool | None = None,
) -> REMLResult:
    """Optimize the direct REML objective via damped Newton (Wood 2011).

    ``warm_lambdas`` (a previous fit's estimates, keyed by component name)
    seeds the bootstrap and the first Newton iterate of every component it
    names; the rest bootstrap cold (``discrete.bootstrap_seed_lambdas``).

    A cold exact fit meeting the pre-solve rule (``_PRESOLVE_*``) first runs
    the cached-W performance iteration on this same design to its fixed
    point, as mgcv's ``outerPIsteps`` initialize the outer iteration, and the
    exact loop starts from its smoothing parameters and coefficients.
    ``presolve`` overrides the rule for tests only.

    ``discrete=False`` runs W(rho)-corrected direct REML; ``discrete=True``
    delegates to ``optimize_discrete_reml_cached_w``, the bam-style cached-W
    optimizer.
    """

    # The declared contract's likelihood size (``lazy_likelihood_size``).
    _contract_size = lazy_likelihood_size(sample_weight, weight_semantics=weight_semantics)

    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    w_correction_order = validate_w_correction_order(w_correction_order)
    if discrete:
        # The cached-W branch holds W fixed within a step (no dW/drho term), so
        # it does not reach the exact optimum; its curvature follows
        # ``classify_binned_reml_curvature``.
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
            weight_semantics=weight_semantics,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            verbose=verbose,
            penalty_caches=penalty_caches,
            profile=profile,
            select_snap=select_snap,
            direct_solve=direct_solve,
            reml_penalties=penalties,
            estimated_names=estimated_names,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
            debug_recorder=debug_recorder,
            trace_run=trace_run,
            warm_lambdas=warm_lambdas,
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
        lambda2=lambdas,
        nesting_cache=getattr(dm, "_structured_layout_cache", None),
    )
    use_structured = structured_decision.use_structured
    record_auto_backend_decision(profile, direct_solve, structured_decision)
    # A structured-capable model builds the same Newton model on the dense
    # solver as a structured factor does, so both backends take one REML path
    # (except binomial/logit, ``_LEVERAGE_HESSIAN_MIN_GRAM_WORK``).
    structured_alternative = (
        select_structured_group(list(dm.group_matrices), groups, mode="auto").group_index
        is not None
    )
    sparse_leverage = not structured_alternative or reads_row_supports(dm)
    if use_observed_geometry:
        validate_observed_derivative_capability(distribution, link, w_correction_order)
    observed_tabmat_state = TabmatCenteringState() if use_observed_geometry else None
    observed_pirls_tol = (
        min(pirls_tol, _OBSERVED_PIRLS_TOL_CEILING) if use_observed_geometry else pirls_tol
    )
    group_names = [pc.name for pc in penalties]
    m = len(group_names)
    # estimated_mask[i] = True  => component i is free to be optimized
    #                     False => component i has a fixed lambda (policy)
    if estimated_names is not None:
        estimated_mask = np.array([pc.name in estimated_names for pc in penalties])
    else:
        estimated_mask = np.ones(m, dtype=bool)
    log_lo, log_hi = np.log(LAMBDA_FLOOR), np.log(LAMBDA_CEILING)
    max_newton_step = 5.0
    _tol = reml_tol
    # Every PIRLS the criterion reads certifies its mode at the bar this
    # stopping tolerance needs (``mode_score``).
    mode_bar = mode_certification_bar(reml_tol)
    # The Laplace approximation's identified part (``reml.identified``):
    # decided once, from the design and prior weights.
    identified = IdentifiedLaplace.for_design(
        dm, sample_weight, penalties, y=y, distribution=distribution, link=link
    )
    # A dense Fisher evaluation's identified part restricts the centred
    # Hessian its PIRLS decomposed; the cache holds it (no extra pass).
    identified_dense = bool(identified) and not use_structured
    # PIRLS stops on the certificate's centred score: a deviance change is
    # blind to a coefficient with a small likelihood share whose curvature
    # enters log|H| at full weight. A linearly constrained mode keeps the
    # deviance stop, certified by its inner QP/KKT state.
    pirls_convergence = (
        "mode_score"
        if use_observed_geometry
        or not any(getattr(group, "constraints", None) is not None for group in groups)
        else "deviance"
    )

    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    objective_history: list[float] = []
    warm_beta: NDArray | None = None
    warm_intercept: float | None = None
    # the warm state's centred predictor (alpha, c), carried with warm_beta and
    # warm_intercept (``mode_score.centred_warm_start``)
    warm_centred: tuple[float, NDArray] | None = None

    best_obj = np.inf
    best_lambdas = lambdas.copy()
    best_pirls = None
    # In-loop fits skip rank metadata (the O(p) geometry summary suffices;
    # the terminal refit publishes); diagnostic runs keep it for the trace.
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
    _n_w_correction = 0
    _t_observed_geometry = 0.0
    _accepted_observed_mode_residual_max = 0.0
    _rejected_trial_observed_mode_residual_max = 0.0
    _observed_mode_rejected_trial_count = 0
    # line-search trials whose structured factor refused (a shorter step)
    _n_refused_structured_trials = 0
    # score-test stops refused because a concave shoulder was live
    _n_concave_stop_refusals = 0
    # solved-row passes forming the W(rho) cross terms at candidate stops (table route)
    _n_stop_cross_passes = 0
    _t_linesearch = 0.0
    _n_linesearch_fits = 0
    # Owned by this optimizer: only a rejected raw-centering route crosses fits.
    raw_moment_policy = TabmatCenteringState()
    fisher_data_reuse = _FisherDataReuse()
    # Owned by this optimizer, dropped with it: the initial systems of fits that start where
    # another ended (``_InitialDataReuse``).
    initial_data_reuse = (
        _InitialDataReuse()
        if not use_observed_geometry and debug_recorder is None and trace_run is None
        else None
    )
    # The observed geometry's centred systems, for the same owner and lifetime.
    geometry_data_reuse = _InitialDataReuse() if use_observed_geometry else None
    # Once a Gamma/log solve leaves Fisher scoring for Newton
    # (``fit_irls_direct``), later solves start on Newton.
    newton_latch = [False]

    # === Pre-solve: cached-W performance iteration on this same design ===
    # The cheap cached-W loop covers the far-from-optimum phase; its fixed
    # point is a start, and the exact loop decides from there.
    presolve_beta: NDArray | None = None
    presolve_intercept: float | None = None
    presolve_centred: tuple[float, NDArray] | None = None
    if presolve is None:
        presolve = (
            not warm_lambdas
            and int(np.count_nonzero(estimated_mask)) >= _PRESOLVE_MIN_ESTIMATED
            and (type(distribution), type(link)) in _PRESOLVE_PAIRS
            and not structured_alternative
            and max_reml_iter >= _PRESOLVE_MIN_REML_ITER
        )
    if presolve and not all_lambdas_fixed:
        presolve_profile: dict = {}
        presolve_start: dict = {}
        _t0 = _time.perf_counter()
        pre = optimize_discrete_reml_cached_w(
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
            weight_semantics=weight_semantics,
            max_reml_iter=max_reml_iter,
            reml_tol=min(reml_tol, _PRESOLVE_REML_TOL),
            verbose=verbose,
            penalty_caches=penalty_caches,
            profile=presolve_profile,
            select_snap=select_snap,
            direct_solve=direct_solve,
            reml_penalties=penalties,
            estimated_names=estimated_names,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
            warm_lambdas=warm_lambdas,
            start_out=presolve_start,
            start_objective_tol=_PRESOLVE_OBJECTIVE_TOL,
        )
        warm_lambdas = {
            **(warm_lambdas or {}),
            **{
                pc.name: float(pre.lambdas[pc.name])
                for pc, free in zip(penalties, estimated_mask, strict=True)
                if free
            },
        }
        presolve_beta = presolve_start["beta"]
        presolve_intercept = presolve_start["intercept"]
        presolve_centred = presolve_start["centred"]
        presolve_wall = _time.perf_counter() - _t0
        if trace_run is not None and trace_run.enabled:
            # The pre-solve runs untraced; one summary event stands for it, and
            # the stream's REML evaluations start after it.
            trace_run.emit_lazy(
                "step_decision",
                lambda: {
                    "solver": "direct_reml",
                    "phase": "presolve",
                    "updates": int(pre.n_reml_iter),
                    "objective": float(pre.objective) if pre.objective is not None else None,
                    "lambdas": dict(pre.lambdas),
                    "wall_s": float(presolve_wall),
                    "termination_reason": pre.termination_reason,
                },
                channel="reml",
                purpose="reml_presolve",
                authoritative=False,
            )
        if profile is not None:
            profile["reml_presolve_s"] = presolve_wall
            profile["reml_presolve_n_iter"] = int(pre.n_reml_iter)
            profile["reml_presolve_converged"] = bool(pre.converged)
            profile["reml_presolve_termination_reason"] = pre.termination_reason
            profile["reml_presolve_objective"] = pre.objective
            for key, value in presolve_profile.items():
                if key.startswith(("irls_", "reml_", "dm_")) and isinstance(value, int | float):
                    profile[f"presolve_{key}"] = value

    # === Bootstrap: one FP step from the seeds ===
    # Seeds from ``bootstrap_seed_lambdas``.
    boot_lambdas = bootstrap_seed_lambdas(
        penalties,
        warm_lambdas,
        lambdas,
        estimated_mask,
        dm.group_matrices,
        distribution=distribution,
        link=link,
        y=y,
        sample_weight=sample_weight,
        offset=offset_arr,
        weight_semantics=weight_semantics,
    )
    # After the pre-solve the first candidate starts from its mode; no bootstrap.
    skip_bootstrap = presolve_beta is not None
    if skip_bootstrap:
        S_boot = None
        boot_result = boot_inv = None
        warm_beta = np.array(presolve_beta, dtype=np.float64, copy=True)
        warm_intercept = presolve_intercept
        warm_centred = presolve_centred
    else:
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
            # Nothing reads the bootstrap's rank metadata, and its data rank can
            # take an O(n p^2) factor: a traced fit records it, as for candidates.
            compute_rank_info=loop_fit_statistics,
            debug_context={"phase": "bootstrap", "reml_iteration": 0},
            trace_run=trace_run,
            trace_purpose="reml_bootstrap",
            _compensate_centred_intercept=False,
            weight_semantics=weight_semantics,
            _initial_data_reuse=initial_data_reuse,
            _raw_moment_policy=raw_moment_policy,
            _fisher_data_reuse=fisher_data_reuse,
            _newton_latch=newton_latch,
            _mode_bar=mode_bar,
        )
        _t_pirls += _time.perf_counter() - _t0
        warm_beta = boot_result.beta.copy()
        warm_intercept = float(boot_result.intercept)
        warm_centred = centred_warm_start(boot_result)

    boot_phi = 1.0
    boot_inv_phi = 1.0
    if not scale_known and not skip_bootstrap:
        assert boot_result is not None
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
                saturated_log_weight=saturated_log_weight or 0.0,
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
        elif isinstance(distribution, Tweedie):
            assert tweedie_scale_data is not None
            boot_scale = profile_tweedie_reml_scale(
                tweedie_scale_data,
                penalized_deviance,
                boot_penalty_nullity,
            )
            boot_phi = boot_scale.phi
            boot_inv_phi = boot_scale.inverse_phi
        else:
            boot_phi = max(
                penalized_deviance / max(_contract_size() - boot_penalty_nullity, 1.0),
                1e-10,
            )
            boot_inv_phi = 1.0 / max(boot_phi, 1e-10)
    bootstrap_log_step_cap = 4.0

    # Store original fixed lambda values so they can be restored exactly
    # after the exp(rho)->clip round-trip (which would clamp 0.0 to LAMBDA_FLOOR).
    fixed_lambdas: dict[str, float] = {}
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_lambdas[pc.name] = float(lambdas[pc.name])

    rho = np.zeros(m, dtype=np.float64)
    for i, pc in enumerate(penalties):
        if not estimated_mask[i]:
            fixed_val = fixed_lambdas[pc.name]
            rho[i] = np.clip(np.log(max(fixed_val, LAMBDA_FLOOR)), log_lo, log_hi)
            continue
        if warm_lambdas and pc.name in warm_lambdas:
            # The warm start is the iterate; no Fellner-Schall step from it.
            rho[i] = np.clip(np.log(boot_lambdas[pc.name]), log_lo, log_hi)
            continue
        # Reached only cold: after the pre-solve every component is warm.
        assert boot_result is not None and boot_inv is not None
        gm = dm.group_matrices[pc.group_index]
        beta_g = boot_result.beta[pc.group_sl]
        quad = penalty_component_quadratic(pc, beta_g, gm)
        trace_term = as_hessian_factor(boot_inv).trace_inverse_penalty(pc)
        r_j = pc.rank if pc.rank > 0 else (penalty_ranks[pc.name] if penalty_ranks else 0.0)
        denom = boot_inv_phi * quad + trace_term
        lam_fp = r_j / denom if denom > 1e-12 else 1.0
        # Snap degenerate select=True null-space penalties to the upper bound:
        # with quad << trace any lambda is nearly a fixed point.
        if (
            select_snap
            and pc.component_type == "selection"
            and trace_term > 1e-12
            and boot_inv_phi * quad < 0.1 * trace_term
        ):
            lam_fp = np.exp(log_hi)
        lam_prev = max(float(lambdas.get(pc.name, 1e-4)), LAMBDA_FLOOR)
        log_prev = np.log(lam_prev)
        log_target = np.log(max(lam_fp, LAMBDA_FLOOR))
        if ":" in pc.group_name:
            log_target = np.clip(
                log_target,
                log_prev - bootstrap_log_step_cap,
                log_prev + bootstrap_log_step_cap,
            )
        rho[i] = np.clip(log_target, log_lo, log_hi)

    def fit_candidate(
        candidate_lambdas: dict[str, float],
        beta_start: NDArray | None,
        intercept_start: float | None,
        iteration: int,
        centred_start: tuple[float, NDArray] | None = None,
    ) -> tuple[Any, Any, Any, NDArray | None, tuple[NDArray, float] | None]:
        """The candidate's penalized mode, inverse, weighted Gram, penalty and dense identified Hessian."""
        nonlocal _t_pirls
        S_fit = (
            None
            if use_structured
            else build_penalty_matrix(
                dm.group_matrices,
                groups,
                candidate_lambdas,
                dm.p,
                reml_penalties=penalties,
            )
        )
        _t_start = _time.perf_counter()
        fit_cache: dict | None = {} if identified_dense else None
        fit_result, fit_inv, fit_xtwx = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=distribution,
            link=link,
            groups=groups,
            lambda2=candidate_lambdas,
            offset=offset_arr,
            beta_init=beta_start,
            intercept_init=intercept_start,
            _centred_init=centred_start,
            max_iter=max_pirls_iter,
            tol=observed_pirls_tol,
            convergence=pirls_convergence,
            return_xtwx=True,
            profile=profile,
            direct_solve=direct_solve,
            S_override=S_fit,
            reml_penalties=penalties,
            compute_rank_info=loop_fit_statistics,
            _compute_fit_statistics=loop_fit_statistics,
            debug_recorder=debug_recorder,
            debug_context={"phase": "candidate", "reml_iteration": iteration},
            trace_run=trace_run,
            trace_purpose="reml_candidate",
            _compensate_centred_intercept=False,
            weight_semantics=weight_semantics,
            _initial_data_reuse=initial_data_reuse,
            _raw_moment_policy=raw_moment_policy,
            _fisher_data_reuse=fisher_data_reuse,
            _newton_latch=newton_latch,
            _laplace_excluded=tuple(int(index) for index in identified.weak),
            _mode_bar=mode_bar,
            cache_out=fit_cache,
        )
        _t_pirls += _time.perf_counter() - _t_start
        return fit_result, fit_inv, fit_xtwx, S_fit, dense_hessian(fit_cache)

    n_mean_space_restorations = 0
    prev_obj = np.inf
    # The stop judges the active set only: a flat frozen direction keeps a
    # tiny gradient forever, and counting it would spin to max_reml_iter.
    stop_criterion_frozen = None
    # The order-1 Hessian omits the d2W terms, so its Newton steps converge
    # only linearly where that curvature is real (Nocedal & Wright 2006, thm
    # 3.6). After two slow reversals in a row (``_slow_reversal``) at a rate
    # that cannot reach the score bar in time, the search takes the exact
    # Hessian (Wood 2011) where the family and link supply it.
    try:
        exact_hessian_available = not discrete and w_correction_order == 1
        if exact_hessian_available:
            validate_observed_derivative_capability(distribution, link, 2)
    except NotImplementedError:
        exact_hessian_available = False
    previous_delta: NDArray | None = None
    previous_gradient_norm = np.inf
    previous_model: tuple[NDArray, NDArray, NDArray, float] | None = None
    step_reversals = 0

    if verbose:
        boot_lam_str = ", ".join(
            f"{name}={np.exp(rho[i]):.4g}" for i, name in enumerate(group_names)
        )
        print(f"  REML bootstrap: lambdas=[{boot_lam_str}]")

    step_bar = newton_step_stop_bar(_tol)
    held_newton_model: tuple[NDArray, NDArray, float] | None = None
    held_cross: NDArray | None = None
    last_accepted_step = 0.0
    for outer in range(max_reml_iter):
        n_iter = outer + 1
        rho_clipped = np.clip(rho, log_lo, log_hi)

        cand_lambdas = lambdas.copy()
        for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
            cand_lambdas[name] = float(np.clip(val, LAMBDA_FLOOR, LAMBDA_CEILING))
        # Restore exact fixed values (exp->clip would clamp 0.0 to LAMBDA_FLOOR)
        cand_lambdas.update(fixed_lambdas)

        if _carry_forward is not None and _carry_forward[0] == cand_lambdas:
            _, pirls_result, XtWX_S_inv, XtWX, S_cand, cand_dense = _carry_forward
            if profile is not None:
                profile["reml_candidate_reuses"] = profile.get("reml_candidate_reuses", 0) + 1
        else:
            pirls_result, XtWX_S_inv, XtWX, S_cand, cand_dense = fit_candidate(
                cand_lambdas, warm_beta, warm_intercept, n_iter, warm_centred
            )
        # No interior mode: the penalized maximum holds rows on the mean-space
        # boundary, where the Laplace approximation does not exist (Wood, Pya &
        # Saefken 2016, sec. 3.1). Raise every estimated penalty in doubling
        # log-steps (Nocedal & Wright 2006, sec. 3.5) until a cold refit is
        # interior; a boundary state is no start. Trials are simply rejected.
        # A binomial/identity boundary state is feasible, and is disclosed instead.
        restoration_step = 1.0
        rho_at_boundary = rho_clipped
        while pirls_result.termination_reason == "mean_space_boundary" and isinstance(
            link, LogLink
        ):
            if not bool(np.any(estimated_mask & (rho_clipped < log_hi))):
                raise ObservedModeNotConvergedError(
                    _mean_space_boundary_message(
                        int(pirls_result.mean_space_boundary_rows),
                        estimated=bool(np.any(estimated_mask)),
                        unresolved=bool(pirls_result.mean_space_boundary_unresolved),
                    ),
                    infeasible_detail=(
                        "the penalized maximum lies on the boundary of the binomial mean space"
                    ),
                )
            rho_clipped = np.where(
                estimated_mask,
                np.clip(rho_at_boundary + restoration_step, log_lo, log_hi),
                rho_at_boundary,
            )
            rho = rho_clipped
            held_newton_model = None
            last_accepted_step = 0.0
            restoration_step *= 2.0
            n_mean_space_restorations += 1
            cand_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
                cand_lambdas[name] = float(np.clip(val, LAMBDA_FLOOR, LAMBDA_CEILING))
            cand_lambdas.update(fixed_lambdas)
            pirls_result, XtWX_S_inv, XtWX, S_cand, cand_dense = fit_candidate(
                cand_lambdas, None, None, n_iter
            )
        warm_beta = pirls_result.beta.copy()
        warm_intercept = float(pirls_result.intercept)
        warm_centred = centred_warm_start(pirls_result)

        geometry: ObservedREMLGeometry | None = None
        # Under observed geometry the geometry's own identified part replaces
        # these below, so the PIRLS factor is not restricted as well.
        reml_inverse = XtWX_S_inv
        objective_logdet = pirls_result.log_det_H
        objective_hessian_rank: int | None = None
        if identified and not use_observed_geometry:
            reml_inverse = identified.inverse(XtWX_S_inv, cand_dense)
            objective_logdet = identified.log_det(XtWX_S_inv, pirls_result.log_det_H, cand_dense)
            objective_hessian_rank = identified.rank(
                pirls_result.reml_hessian_rank
                if pirls_result.reml_hessian_rank is not None
                else 1 + dm.p,
                XtWX_S_inv,
                cand_dense,
            )
        # Stationarity is PIRLS's own stop: the mode certificate, or the
        # deviance stop for a linearly constrained mode. The observed
        # geometry's mode score below is a diagnostic only.
        candidate_mode_stationary = bool(pirls_result.converged)
        if use_observed_geometry:
            if not pirls_result.converged and not stopped_on_iteration_budget(pirls_result, link):
                # Typed so a power search routes around the point. Reasons
                # here are structural failures, never budget exhaustion.
                raise ObservedModeNotConvergedError(hint=mode_certification_hint(distribution))
            _t0 = _time.perf_counter()
            try:
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
                    _data_reuse=geometry_data_reuse,
                    derivative_order=w_correction_order,
                    groups=groups if use_structured else None,
                    lambdas=cand_lambdas if use_structured else None,
                    reml_penalties=penalties if use_structured else None,
                    structured_group_index=(
                        structured_decision.group_index if use_structured else None
                    ),
                    structured_chain_group_indices=(
                        structured_decision.chain_group_indices if use_structured else ()
                    ),
                )
            except ObservedGeometryInfeasibleError as exc:
                # An infeasible geometry has no step to halve here, so retype it
                # for power searches to route around. The build's other
                # ValueErrors are caller-contract bugs and stay bare.
                _t_observed_geometry += _time.perf_counter() - _t0
                raise ObservedModeNotConvergedError(
                    "observed REML geometry refused the penalized coefficient mode "
                    f"at this point: {exc}",
                    hint=mode_certification_hint(distribution),
                    infeasible_detail="observed geometry refused the penalized mode",
                ) from exc
            geometry = identified.geometry(geometry)
            _t_observed_geometry += _time.perf_counter() - _t0
            if geometry.hessian_inverse is None:  # pragma: no cover - requested above
                raise RuntimeError("observed REML geometry omitted its slope inverse")
            reml_inverse = geometry.hessian_inverse
            objective_logdet = geometry.log_det_H
            objective_hessian_rank = geometry.hessian_rank
            try:
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
                    excluded=identified.weak,
                    bar=mode_bar,
                )
            except ObservedGeometryInfeasibleError as exc:
                # An unusable score gets the same retype as the geometry above;
                # the score's other ValueErrors are caller bugs and stay bare.
                raise ObservedModeNotConvergedError(
                    "observed REML mode score refused the penalized coefficient mode "
                    f"at this point: {exc}",
                    hint=mode_certification_hint(distribution),
                    infeasible_detail="observed mode score refused the penalized mode",
                ) from exc
            _accepted_observed_mode_residual_max = max(
                _accepted_observed_mode_residual_max,
                mode_score.relative_max,
            )
            # A mode that misses the bar is never refused: it cannot authorise a
            # convergence exit, so a fit that never certifies ends not converged
            # and says so. The geometry's score is kept as a diagnostic.

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
        obj_roundoff = 0.0
        if isinstance(objective_evaluation, REMLObjectiveEvaluation):
            obj = objective_evaluation.value
            obj_roundoff = objective_evaluation.roundoff
            penalty_nullity = objective_evaluation.penalty_nullity
            profiled_scale = objective_evaluation.profiled_scale
        else:
            # A test or instrumentation stub may return a bare scalar.
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
                    penalized_deviance / max(_contract_size() - penalty_nullity, 1.0),
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
                profile["reml_freeze_decision"] = fixed_freeze_record(
                    group_names,
                    direction_penalty_ranks(penalties, penalty_ranks),
                    objective=obj,
                    estimated_mask=estimated_mask,
                )
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
        leverage_hessian = False
        binomial_logit = False
        w_correction_args: tuple = ()
        w_correction_kwargs: dict[str, Any] = {}
        if not discrete:
            # Binomial/logit on a dense factor takes the leverage route at any
            # work, with the exact gradient and no W(rho) cross terms
            # (``gradient_terms_only``), at every iteration and at the stop:
            # its derivative Grams cost several leverage passes.
            leverage_ready = w_correction_order == 1 and len(penalties) >= _LEVERAGE_MIN_PENALTIES
            binomial_logit = bool(
                leverage_ready
                and type(distribution) is Binomial
                and type(link) is LogitLink
                and geometry is None
                and isinstance(as_hessian_factor(reml_inverse), DenseHessianFactor)
            )
            leverage_hessian = bool(
                binomial_logit
                or (
                    leverage_ready
                    and float(len(penalties)) * dm.n * dm.p * dm.p
                    >= _LEVERAGE_HESSIAN_MIN_GRAM_WORK
                    and sparse_leverage
                )
            )
            # A LinAlgError from the correction (a compact operator refusing
            # its blocks) means no usable curvature here, so it is retyped;
            # structural ValueError/TypeError propagate bare. Not gated on
            # `use_observed_geometry`: Tweedie + PowerLink turns Fisher at the
            # canonical power, and the same failure must stay routable there.
            w_correction_args = (dm, link, groups, pirls_result, reml_inverse, cand_lambdas)
            w_correction_kwargs = {
                "penalty_caches": penalty_caches,
                "sample_weight": sample_weight,
                "offset_arr": offset_arr,
                "distribution": distribution,
                "w_correction_order": w_correction_order,
                "reml_penalties": penalties,
                "geometry": geometry,
                "leverage_hessian": leverage_hessian,
                "gradient_terms_only": structured_alternative or binomial_logit,
            }
            w_corr = _usable_w_correction(
                reml_w_correction, w_correction_args, w_correction_kwargs, distribution
            )
        else:
            w_corr = None
        if profile is not None:
            correction_mode = "none"
            if w_corr is not None:
                if w_corr[1] is not None:
                    correction_mode = "derivative_grams"
                else:
                    # The leverage route's last entry: its cross terms, or None
                    # when its Newton model takes the gradient alone.
                    correction_mode = (
                        "leverage_hessian" if w_corr[-1] is not None else "leverage_tables"
                    )
            profile["reml_w_correction_mode"] = correction_mode
        _t_w_correction += _time.perf_counter() - _t0
        if w_corr is not None:
            _n_w_correction += 1
            grad_w_correction = w_corr[0]
            dH_extra = w_corr[1]
            # Second order's d2W terms, or the leverage route's first-order cross
            # terms: either is an (m, m) term added to the Hessian.
            dH2_cross = w_corr[2] if len(w_corr) > 2 else None
            grad = grad_partial + grad_w_correction
        else:
            grad = grad_partial.copy()
            dH_extra = None
            dH2_cross = None
        # The leverage route's gradient-only Newton model forms its W(rho)
        # cross terms at the candidate stop below (binomial/logit excepted).
        table_route = (
            leverage_hessian
            and not binomial_logit
            and w_corr is not None
            and len(w_corr) > 2
            and w_corr[1] is None
            and w_corr[2] is None
        )

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

        # Wood (2011) eq. 6.2's diagonal term g_i + 0.5 r_i takes the fixed-W
        # gradient: the W(rho) terms enter through dH_extra and dH2_cross, and
        # the total gradient would count them twice.
        def outer_hessian() -> NDArray:
            return reml_direct_hessian(
                dm.group_matrices,
                distribution,
                reml_inverse,
                cand_lambdas,
                gradient=grad_partial,
                penalty_caches=penalty_caches,
                pirls_result=pirls_result,
                n_obs=_contract_size(),
                phi_hat=phi_hat,
                inverse_phi=inverse_phi,
                d_inverse_phi_d_penalized_deviance=inverse_phi_derivative,
                penalty_nullity=penalty_nullity if not scale_known else None,
                dH_extra=dH_extra,
                dH2_cross=dH2_cross,
                reml_penalties=penalties,
            )

        def stop_cross_terms() -> NDArray | None:
            """The W(rho) cross terms a table-route Newton model omits, by one solved-row pass."""
            nonlocal _n_stop_cross_passes, _t_w_correction
            started = _time.perf_counter()
            try:
                return _usable_w_correction(
                    reml_w_correction,
                    w_correction_args,
                    {**w_correction_kwargs, "lookup_tables": False},
                    distribution,
                )[2]
            finally:
                # A failed confirmation still paid for its attempted pass.
                _n_stop_cross_passes += 1
                _t_w_correction += _time.perf_counter() - started

        def held_step_left() -> float:
            """The Newton step this gradient leaves on the held Hessian, infinite if unconfirmed.

            The cross-term pass is evidence for an optional stop, so its failure
            reads as not confirmed, and the search goes on.
            """
            nonlocal held_cross
            hessian, index, _ = held_newton_model
            if table_route:
                try:
                    held_cross = stop_cross_terms()
                except ObservedModeNotConvergedError:
                    return np.inf
            if held_cross is not None:
                hessian = hessian + held_cross[np.ix_(index, index)]
            return newton_step_left(hessian, grad[index])

        def record_step_stop(step_left: float, *, step_taken: bool) -> None:
            if profile is not None:
                profile["reml_newton_step_stop"] = {
                    "iter": int(n_iter),
                    "step_left": step_left,
                    "taken_step": held_newton_model[2] if step_taken else None,
                    "previous_step": held_newton_model[2] if step_taken else last_accepted_step,
                    "bar": step_bar,
                    "step_taken": step_taken,
                }

        revalidated_hess = None
        # An uncertified candidate's objective and gradient prove nothing, so
        # the stop waits until the mode certifies (warm-started settles).
        mode_certified = trial_counts_as_precision_evidence(candidate_mode_stationary, obj)
        if candidate_convergence.converged and mode_certified:
            # The gradient-only Newton model omits -(P + P')/2, whose diagonal
            # has no fixed sign, so the stop's second-order check and freeze read
            # a Hessian with the cross terms, formed here by one solved-row pass.
            # A refused stop steps on that Hessian too, so its freeze keeps the
            # directions the check reactivated.
            stop_cross = stop_cross_terms() if table_route else None
            _t0 = _time.perf_counter()
            revalidated_hess = outer_hessian()
            if stop_cross is not None:
                revalidated_hess = revalidated_hess + stop_cross
            stop, refused = revalidate_stop(
                proj_grad,
                revalidated_hess,
                stop_criterion_frozen,
                estimated_mask,
                objective=obj,
                tolerance=_tol,
                score_scale=score_scale,
                step_bound=max_newton_step,
                names=group_names,
                iteration=n_iter,
                profile=profile,
                concave_live=concave_live_directions,
                freeze=freeze_flat_directions,
                ranks=lambda: direction_penalty_ranks(penalties, penalty_ranks),
            )
            _n_concave_stop_refusals += refused
            _t_hessian += _time.perf_counter() - _t0
            if stop:
                converged = True
                termination_reason = "score_objective_tolerance"
                break
            # A re-activated direction with a live gradient: not converged.
            # Fall through into this iteration's Newton step, reusing the
            # Hessian just computed (with the stop's cross terms on the table
            # route).
        elif (
            held_newton_model is not None
            and mode_certified
            and not np.any(stop_grad[stop_criterion_frozen])
            and newton_step_settled(
                step_left := held_step_left(), held_newton_model[2], bar=step_bar
            )
        ):
            record_step_stop(step_left, step_taken=True)
            converged = True
            termination_reason = "newton_step_tolerance"
            break
        held_newton_model = None
        if held_cross is not None:
            # A refused stop steps on the Hessian with the cross terms it paid for.
            _t0 = _time.perf_counter()
            revalidated_hess = outer_hessian() + held_cross
            held_cross = None
            _t_hessian += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        hess = revalidated_hess if revalidated_hess is not None else outer_hessian()
        if use_structured and not (np.all(np.isfinite(hess)) and np.all(np.isfinite(grad))):
            # Prior weights near float64's range edge can make the trace terms
            # non-finite; no Newton step exists, so raise a clear error.
            raise StructuredSolverError(
                "The structured solver cannot proceed: the REML outer gradient or Hessian "
                "is not finite at these smoothing parameters. This should not happen for a "
                "model its data identify; direct_solve='gram' fits it with the dense solver "
                "instead."
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
            step_bound=max_newton_step,
        )
        frozen = freeze_decision.frozen
        shoulder = np.asarray(freeze_decision.shoulder, dtype=bool)
        if outer == 0 and presolve_beta is not None:
            # The pre-solve's fixed point lacks the dW/drho term, so no direction
            # is yet shown flat on the exact criterion: the first exact step
            # moves every direction whose curvature puts a minimum within the
            # step bound. Directions at a bound or with none in reach stay held.
            diagonal = np.diagonal(hess)
            in_reach = (
                estimated_mask
                & (diagonal > 0.0)
                & (proj_grad != 0.0)
                & (np.abs(proj_grad) <= max_newton_step * diagonal)
            )
            frozen = frozen & ~in_reach
        active_idx = np.where(~frozen)[0]
        stop_criterion_frozen = frozen.copy()
        if profile is not None:
            profile["reml_freeze_decision"] = freeze_record(
                group_names,
                proj_grad,
                hess,
                freeze_decision,
                score_scale=score_scale,
                estimated_mask=estimated_mask,
                frozen=frozen,
            )
            if shoulder.any():
                # Every iteration that steps a shoulder, with its signed
                # gradient and curvature.
                idx = np.flatnonzero(shoulder)
                profile.setdefault("reml_shoulder_reads", []).append(
                    {
                        "iter": int(n_iter),
                        "names": [group_names[i] for i in idx],
                        "gradient": [float(proj_grad[i]) for i in idx],
                        "curvature": [float(hess[i, i]) for i in idx],
                    }
                )

        if active_idx.size == 0:
            rho = rho_clipped
            _t_hessian += _time.perf_counter() - _t0
            if outer == 0:
                # Do not let a deliberately loose tolerance bypass the
                # two-evaluation convergence contract.
                continue
            if not (
                trial_counts_as_precision_evidence(candidate_mode_stationary, obj)
                and obj_change < _tol * score_scale
            ):
                # An all-frozen verdict at a nonstationary candidate defers to a
                # warm-started settle at the same lambdas.
                continue
            # All components frozen: converged (a genuinely all-flat model; the
            # compound criterion usually fires first).
            converged = True
            termination_reason = "active_set_stationary"
            break

        # Modified Newton: eigendecompose, flip negatives, floor small eigenvalues.
        # A concave shoulder instead steps to the trust-region boundary (the
        # step cap) downhill (More & Sorensen 1979).
        newton_idx = np.where(~frozen & ~shoulder)[0] if shoulder.any() else active_idx
        delta = np.zeros(m)
        if newton_idx.size:
            if newton_idx.size < m:
                hess_sub = hess[np.ix_(newton_idx, newton_idx)]
                grad_sub = grad[newton_idx]
            else:
                hess_sub = hess
                grad_sub = grad

            eigvals_h, eigvecs_h = np.linalg.eigh(hess_sub)
            max_eig = max(abs(eigvals_h).max(), 1e-12)
            eig_floor = max_eig * NEWTON_EIGENVALUE_FLOOR
            eigvals_pd = np.maximum(np.abs(eigvals_h), eig_floor)

            hess_pd = (eigvecs_h * eigvals_pd) @ eigvecs_h.T
            delta_sub = -np.linalg.solve(hess_pd, grad_sub)

            # Scatter back to full delta
            delta[newton_idx] = delta_sub
        delta[shoulder] = -np.sign(proj_grad[shoulder]) * max_newton_step

        # Proportional step cap: scale entire vector if any component > max_step
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_newton_step:
            delta *= max_newton_step / max_delta
        # A negligible Newton step is still taken, since it is already paid for,
        # and this Hessian then confirms it at the next candidate.
        if (
            newton_idx.size
            and not shoulder.any()
            and mode_certified
            and revalidated_hess is None
            and max_delta <= step_bar
        ):
            held_hessian = hess[np.ix_(newton_idx, newton_idx)]
            if newton_step_resolved(held_hessian, grad[newton_idx], bar=step_bar):
                held_newton_model = (held_hessian, newton_idx, max_delta)
        if exact_hessian_available:
            contraction = (
                proj_grad_norm / previous_gradient_norm if previous_gradient_norm > 0.0 else np.inf
            )
            excess_decrease = np.inf
            if previous_model is not None:
                rho_before, grad_before, hess_before, obj_before = previous_model
                taken = rho_clipped - rho_before
                predicted = -float(grad_before @ taken + 0.5 * taken @ hess_before @ taken)
                excess_decrease = (obj_before - obj) - 2.0 * predicted
            previous_model = (rho_clipped.copy(), grad.copy(), hess.copy(), obj)
            slow_reversal = _slow_reversal(
                delta, previous_delta, contraction, excess_decrease, obj_change, obj, _tol
            )
            step_reversals = step_reversals + 1 if slow_reversal else 0
            previous_delta = delta.copy()
            previous_gradient_norm = proj_grad_norm
            if _escalates_to_exact_hessian(
                step_reversals,
                slow_reversal,
                _tol,
                score_scale,
                proj_grad_norm,
                contraction,
                max_reml_iter - n_iter,
            ):
                w_correction_order = 2
                exact_hessian_available = False
                held_newton_model = None
        _t_hessian += _time.perf_counter() - _t0
        # What the dead-search exit below judges: the active set's gradient and
        # the decrease its Newton model predicts (half the squared decrement of
        # the uncapped step, its Hessian unmodified).
        active_grad_norm = (
            float(np.max(np.abs(np.where(frozen, 0.0, proj_grad)))) if proj_grad.size else 0.0
        )
        dead_predicted_decrease = (
            None
            if max_delta > max_newton_step or bool(shoulder.any())
            else newton_predicted_decrease(
                grad_sub, eigvals_h, eigvecs_h, eigenvalue_floor=eig_floor
            )
        )

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
        # an accepted trial's decrease within the two evaluations' rounding,
        # and that rounding
        unresolved_decrease: float | None = None
        stall_roundoff: float | None = None
        # whether a stationary full-step trial was rejected, and an accepted
        # shorter step's decrease that no quadratic criterion rejecting it
        # allows
        full_step_rejected = False
        inconsistent_decrease: float | None = None
        for _ls in range(max_ls):
            rho_trial = np.clip(rho_clipped + step * delta, log_lo, log_hi)
            if np.all(np.abs(rho_trial - rho_clipped) <= 1e-12):
                break
            had_feasible_trial = True
            trial_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_trial), strict=False):
                trial_lambdas[name] = float(np.clip(val, LAMBDA_FLOOR, LAMBDA_CEILING))
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
            trial_cache: dict | None = {} if identified_dense else None
            trial_logdet: float | None = None
            trial_hessian_rank: int | None = None
            try:
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
                    _centred_init=warm_centred,
                    max_iter=max_pirls_iter,
                    tol=observed_pirls_tol,
                    convergence=pirls_convergence,
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
                    _compensate_centred_intercept=False,
                    weight_semantics=weight_semantics,
                    _initial_data_reuse=initial_data_reuse,
                    _raw_moment_policy=raw_moment_policy,
                    _fisher_data_reuse=fisher_data_reuse,
                    _newton_latch=newton_latch,
                    _laplace_excluded=tuple(int(index) for index in identified.weak),
                    _mode_bar=mode_bar,
                    cache_out=trial_cache,
                )
                trial_dense = dense_hessian(trial_cache)
                trial_logdet = trial_result.log_det_H
                if identified and not use_observed_geometry:
                    trial_logdet = identified.log_det(trial_inv, trial_logdet, trial_dense)
                    trial_hessian_rank = identified.rank(
                        trial_result.reml_hessian_rank
                        if trial_result.reml_hessian_rank is not None
                        else 1 + dm.p,
                        trial_inv,
                        trial_dense,
                    )
            except StructuredSolverError:
                # A trial the structured factor refuses has no objective: try a
                # shorter step. No other solver; a refused candidate still raises.
                _n_refused_structured_trials += 1
                step *= 0.5
                continue
            trial_mode_residual: float | None = None
            # Same standard as the candidate above: the certificate, where one
            # runs, is what establishes stationarity.
            trial_mode_stationary = bool(trial_result.converged)
            if use_observed_geometry:
                if not trial_result.converged and not stopped_on_iteration_budget(
                    trial_result, link
                ):
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
                        _data_reuse=geometry_data_reuse,
                        # the identified part rebuilds the structured factor returned here
                        compute_inverse=bool(identified),
                        groups=groups if use_structured else None,
                        lambdas=trial_lambdas if use_structured else None,
                        reml_penalties=penalties if use_structured else None,
                        structured_group_index=(
                            structured_decision.group_index if use_structured else None
                        ),
                        structured_chain_group_indices=(
                            structured_decision.chain_group_indices if use_structured else ()
                        ),
                    )
                except ObservedGeometryInfeasibleError:
                    # An infeasible trial geometry is a line-search rejection;
                    # caller-contract errors do not improve at half the step.
                    _t_observed_geometry += _time.perf_counter() - _t_geometry
                    _observed_mode_rejected_trial_count += 1
                    step *= 0.5
                    continue
                try:
                    trial_geometry = identified.geometry(trial_geometry)
                except StructuredSolverError:
                    # the factor refused its identified part: the same
                    # rejected trial as a refused PIRLS build above
                    _t_observed_geometry += _time.perf_counter() - _t_geometry
                    _n_refused_structured_trials += 1
                    step *= 0.5
                    continue
                _t_observed_geometry += _time.perf_counter() - _t_geometry
                trial_logdet = trial_geometry.log_det_H
                trial_hessian_rank = trial_geometry.hessian_rank
                try:
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
                        excluded=identified.weak,
                        bar=mode_bar,
                    )
                except ObservedGeometryInfeasibleError:
                    # An unusable trial score is a rejection like the geometry
                    # above, with no residual to record; other ValueErrors stay bare.
                    _observed_mode_rejected_trial_count += 1
                    step *= 0.5
                    continue
                trial_mode_residual = trial_mode_score.relative_max
                if not trial_mode_stationary:
                    # PIRLS's own certificate (above) missed the bar: an
                    # uncertified trial is no evidence for the line search.
                    _observed_mode_rejected_trial_count += 1
                    _rejected_trial_observed_mode_residual_max = max(
                        _rejected_trial_observed_mode_residual_max,
                        trial_mode_score.relative_max,
                    )
                    step *= 0.5
                    continue

            try:
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
                    saturated_log_weight=saturated_log_weight,
                    weight_semantics=weight_semantics,
                    gamma_scale_data=gamma_scale_data,
                    tweedie_scale_data=tweedie_scale_data,
                    return_evaluation=True,
                )
            except PenaltyNumericalError:
                # A refused trial has no objective: try a shorter step.
                step *= 0.5
                continue
            trial_obj = (
                trial_evaluation.value
                if isinstance(trial_evaluation, REMLObjectiveEvaluation)
                else float(trial_evaluation)
            )
            if trial_counts_as_precision_evidence(trial_mode_stationary, trial_obj):
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
            trial_roundoff = (
                trial_evaluation.roundoff
                if isinstance(trial_evaluation, REMLObjectiveEvaluation)
                else 0.0
            )
            if (
                not trial_accepted
                and step == 1.0
                and trial_counts_as_precision_evidence(trial_mode_stationary, trial_obj)
            ):
                full_step_rejected = True
            if (
                trial_accepted
                and full_step_rejected
                and dead_predicted_decrease is not None
                and dead_predicted_decrease < _tol * score_scale
                and 2.0 * dead_predicted_decrease * step * (1.0 - (1.0 - armijo_c) * step)
                < obj - trial_obj
                <= _tol * score_scale
                and _backtracked_rate_cannot_finish(
                    active_grad_norm, _tol * score_scale, step, max_reml_iter - n_iter
                )
                and classify_dead_feasible_exit(
                    active_grad_norm,
                    objective=obj,
                    tolerance=_tol,
                    evaluated_trial=evaluated_feasible_trial
                    and trial_counts_as_precision_evidence(candidate_mode_stationary, obj),
                    predicted_decrease=dead_predicted_decrease,
                )
                == "converged_at_precision"
            ):
                # With ``g' delta = -2 P``, a quadratic criterion that rejected
                # the full step decreases by at most ``2 P alpha (1 - (1 - c)
                # alpha)`` at ``alpha``; a larger accepted decrease, within the
                # stop resolution, is evaluation error. Where ``P`` is below that
                # resolution, the dead-search certificate holds and the linear
                # rate cannot reach the score bar in time, the step is discarded
                # and the search ends as a dead one (Shi, Xie, Byrd & Nocedal
                # 2022, SIAM J. Optim. 32(1), sec. 4.1).
                inconsistent_decrease = float(obj - trial_obj)
                if trace_run is not None and trace_run.enabled:
                    # The trial's evaluation event above reads accepted; this
                    # records that the stop discarded it.
                    trace_run.emit_lazy(
                        "step_decision",
                        lambda: {
                            "solver": "direct_reml",
                            "phase": "line_search",
                            "outer_iteration": n_iter,
                            "line_search_iteration": _ls + 1,
                            "trial_alpha": float(step),
                            "inconsistent_decrease": inconsistent_decrease,
                            "predicted_decrease": dead_predicted_decrease,
                        },
                        channel="reml",
                        purpose="reml_inconsistent_decrease",
                        authoritative=False,
                    )
                break
            if (
                trial_accepted
                and presolve_beta is not None
                and obj - trial_obj <= obj_roundoff + trial_roundoff
                and classify_dead_feasible_exit(
                    active_grad_norm,
                    objective=obj,
                    tolerance=_tol,
                    evaluated_trial=evaluated_feasible_trial
                    and trial_counts_as_precision_evidence(candidate_mode_stationary, obj),
                    predicted_decrease=dead_predicted_decrease,
                )
                == "converged_at_precision"
            ):
                # From the pre-solve's start, a decrease within the two
                # evaluations' rounding is no decrease: the dead-search
                # certificate judges the candidate, else the step is taken.
                unresolved_decrease = float(obj - trial_obj)
                stall_roundoff = float(obj_roundoff + trial_roundoff)
                if trace_run is not None and trace_run.enabled:
                    # The trial's evaluation event above reads accepted; this
                    # records that the stop discarded it.
                    trace_run.emit_lazy(
                        "step_decision",
                        lambda: {
                            "solver": "direct_reml",
                            "phase": "line_search",
                            "outer_iteration": n_iter,
                            "line_search_iteration": _ls + 1,
                            "unresolved_decrease": unresolved_decrease,
                            "objective_roundoff": stall_roundoff,
                        },
                        channel="reml",
                        purpose="reml_rounding_stall",
                        authoritative=False,
                    )
                break

            if trial_accepted:
                if trial_mode_residual is not None:
                    _accepted_observed_mode_residual_max = max(
                        _accepted_observed_mode_residual_max,
                        trial_mode_residual,
                    )
                # A shoulder's trust-region move gives no Newton contraction rate.
                last_accepted_step = (
                    0.0 if shoulder.any() else float(np.max(np.abs(rho_trial - rho_clipped)))
                )
                rho = rho_trial
                warm_beta = trial_result.beta.copy()
                warm_intercept = float(trial_result.intercept)
                warm_centred = centred_warm_start(trial_result)
                # The accepted trial is a valid retained candidate even on the
                # final iteration, so ``max_reml_iter`` keeps its improvement.
                if trial_obj < best_obj:
                    best_obj = float(trial_obj)
                    best_lambdas = trial_lambdas.copy()
                    best_pirls = trial_result
                    if outer == max_reml_iter - 1:
                        lambda_history.append(trial_lambdas.copy())
                        objective_history.append(float(trial_obj))
                # Carry a stationary accepted state forward as the next candidate
                # (same penalised system); a nonstationary one is refitted.
                if trial_mode_stationary:
                    _carry_forward = (
                        trial_lambdas.copy(),
                        trial_result,
                        trial_inv,
                        trial_xtwx,
                        S_trial,
                        trial_dense,
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
                # Every projected coordinate is at a bound: re-evaluate once so
                # the compound criterion can confirm stability.
                continue
            # Every feasible trial was rejected. The optimum is resolved when
            # every gradient of the current active set is below the precision
            # asked for (never tighter than the achievable floor), or within a
            # decade of it with the active set's Newton model predicting a
            # decrease below the stop resolution (#459); otherwise
            # line_search_failed (``classify_dead_feasible_exit``).
            dead_evidence = evaluated_feasible_trial and trial_counts_as_precision_evidence(
                candidate_mode_stationary, obj
            )
            termination_reason = classify_dead_feasible_exit(
                active_grad_norm,
                objective=obj,
                tolerance=_tol,
                evaluated_trial=dead_evidence,
                predicted_decrease=dead_predicted_decrease,
            )
            if termination_reason == "line_search_failed" and held_newton_model is not None:
                # The evidence is the candidate's own: a certified mode, a positive
                # definite Hessian and a negligible step. Trials at this floor, refused
                # or rejected, add none either way, so the point stands.
                step_left = held_step_left()
                if newton_step_settled(step_left, last_accepted_step, bar=step_bar):
                    record_step_stop(step_left, step_taken=False)
                    termination_reason = "newton_step_tolerance"
            if profile is not None:
                # The dead-search exit's evidence, as the discrete engine
                # records it.
                profile["reml_dead_line_search"] = {
                    "iter": int(n_iter),
                    "active_gradient_norm": active_grad_norm,
                    "bar": float(max(FLAT_DIRECTION_FREEZE_FLOOR, _tol) * score_scale),
                    "predicted_decrease": dead_predicted_decrease,
                    "decrease_bar": float(_tol * score_scale),
                    "evaluated_trial": bool(dead_evidence),
                    "candidate_mode_stationary": bool(candidate_mode_stationary),
                    "termination_reason": termination_reason,
                    # an accepted trial's decrease, within the two evaluations'
                    # rounding; without one, the candidate's own rounding
                    "unresolved_decrease": unresolved_decrease,
                    "objective_roundoff": (
                        float(obj_roundoff) if stall_roundoff is None else stall_roundoff
                    ),
                    # a discarded shorter step's decrease, past what the rejected
                    # full step leaves a quadratic criterion and within the
                    # stop resolution
                    "inconsistent_decrease": inconsistent_decrease,
                }
            # The second-order check (``concave_live_directions``) gates the
            # score-test exit only: a shoulder here has its gradient under the
            # certified floor, and the rejected trials have just failed to show
            # the decrease its model promises.
            converged = termination_reason in ("converged_at_precision", "newton_step_tolerance")
            break
        _t_linesearch += _time.perf_counter() - _t0

    if best_pirls is None:
        raise RuntimeError("Direct REML Newton did not evaluate any candidates")

    if profile is not None:
        profile["reml_optimizer_s"] = _time.perf_counter() - _t_reml_start
        profile["reml_pirls_s"] = _t_pirls
        profile["reml_objective_s"] = _t_objective
        profile["reml_gradient_s"] = _t_gradient
        profile["reml_w_correction_s"] = _t_w_correction
        profile["reml_n_w_correction"] = _n_w_correction
        profile["reml_observed_geometry_s"] = _t_observed_geometry
        profile["reml_observed_mode_residual_accepted_max"] = _accepted_observed_mode_residual_max
        profile["reml_observed_mode_residual_rejected_trial_max"] = (
            _rejected_trial_observed_mode_residual_max
        )
        profile["reml_observed_mode_rejected_trial_count"] = _observed_mode_rejected_trial_count
        profile["reml_n_refused_structured_trials"] = _n_refused_structured_trials
        profile["reml_n_concave_stop_refusals"] = _n_concave_stop_refusals
        profile["reml_n_stop_cross_passes"] = _n_stop_cross_passes
        profile["reml_w_correction_order"] = int(w_correction_order)
        profile["reml_hessian_newton_s"] = _t_hessian
        profile["reml_linesearch_s"] = _t_linesearch
        profile["reml_n_linesearch_fits"] = _n_linesearch_fits
        profile["reml_n_outer_iter"] = n_iter
        profile["reml_mean_space_restorations"] = n_mean_space_restorations
        profile["reml_laplace_excluded"] = identified.disclosed
        profile["reml_laplace_exclusion_unsupported"] = int(identified.unsupported)

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
        tweedie_scale_data=tweedie_scale_data,
        mean_space_restorations=n_mean_space_restorations,
    )
