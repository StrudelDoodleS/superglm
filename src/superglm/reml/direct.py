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
from superglm.distributions import Binomial, Gamma, Gaussian, Poisson, Tweedie
from superglm.group_matrix import DesignMatrix
from superglm.links import LogitLink, LogLink
from superglm.reml.convergence import (
    FLAT_DIRECTION_FREEZE_FLOOR,
    classify_dead_feasible_exit,
    concave_live_directions,
    direction_penalty_ranks,
    evaluate_reml_candidate,
    freeze_flat_directions,
    mask_frozen_stop_gradient,
    newton_predicted_decrease,
    project_reml_gradient,
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
from superglm.solvers.centered_system import (
    TabmatCenteringState,
    _FisherDataReuse,
    _InitialDataReuse,
)
from superglm.solvers.hessian_factor import DenseHessianFactor, as_hessian_factor
from superglm.solvers.irls_direct import StructuredSolverError, fit_irls_direct
from superglm.solvers.mode_score import centred_warm_start, mode_certification_bar
from superglm.solvers.structured import (
    record_auto_backend_decision,
    resolve_structured_backend,
    select_structured_group,
)
from superglm.types import GroupSlice, PenaltyComponent

# The solve ceiling and the certification bar both live in observed_geometry
# now, shared with the terminal publication gate in model/reml_finalize.py so
# the two can never drift apart.
_OBSERVED_PIRLS_TOL_CEILING = OBSERVED_PIRLS_TOL_CEILING

# One dense leverage pass replaces one derivative Gram per penalty. Mixed
# spline/categorical timings cross over at four penalties on the reference host;
# below that, the specialized Gram route remains cheaper.
_LEVERAGE_GRADIENT_MIN_PENALTIES = 4
# The pre-solve (``presolve``) hands the exact loop a start, not a result: it
# stops once a working-model update moves the objective by less than this
# fraction of its scale.  On freMTPL2 (1 to 10 interaction pairs) the cached-W
# objective then oscillates at 1e-9 of its scale for several more updates
# without moving the exact loop's iteration count; at 1e-6 it stopped early
# enough on 3 pairs to cost three more exact iterations.
_PRESOLVE_OBJECTIVE_TOL = 1e-7
# The pre-solve's stopping tolerance is never looser than this (a tighter
# caller tolerance passes through): its freeze and stop bars scale with it,
# and a loose caller tolerance (1e-5) took the cached-W loop 20 updates where
# 1e-9 took 16 on freMTPL2 (3 pairs), for a start the exact loop then
# corrects anyway.
_PRESOLVE_REML_TOL = 1e-9
# The automatic pre-solve (``presolve=None``) runs on a cold exact fit with at
# least this many estimated smoothing parameters.  Each exact iteration pays
# a weight-correction pass whose cost grows with that count; measured
# single-threaded, off/on wall: 0.75 .. 0.96 with 2 parameters, 0.81 .. 1.14
# with 6 (n = 1,800 / 20,000), 3.0 with 10 (n = 3,000 / 30,000).
_PRESOLVE_MIN_ESTIMATED = 8
# Nor under a short iteration cap: the pre-solve takes up to the fit's own
# ``max_reml_iter`` cached-W updates before the exact loop (7 to 16 measured
# on freMTPL2 and the test fixtures), so a caller capping the search below
# this many iterations -- interaction_mode="fast_candidate" caps it at 5 --
# would get up to twice the work it budgeted for.
_PRESOLVE_MIN_REML_ITER = 10
# It also runs only where it was measured: Poisson with the log link, with
# no RandomEffect or FactorSmooth term the structured solvers could take.
# Elsewhere it would choose its start on curvature the exact loop does not
# use (observed-curvature families), carry a boundary coefficient into the
# exact fit without the mean-space restoration (binomial with the log link),
# take a backend route whose parity with the dense one is untested past a few
# smoothing parameters, or rest on an unmeasured cost: binomial with the logit
# link takes its own leverage-gradient route, and a constant-weight Gaussian
# identity fit's cached-W criterion is already the exact one.  Each of those
# keeps the plain cold start until it is measured.  A Tweedie power
# search turns it off itself: a changed start moves the selected power.
# The other first-order fits keep the derivative Grams, and with them the
# outer Hessian's weight-product term, until those Grams cost real time: one
# ``p x p`` Gram per smoothing parameter is ``m n p^2`` multiply-adds per REML
# iteration against ``n p^2`` for the one-pass route that forms the gradient and
# the first-order Hessian terms (``reml_w_correction(leverage_hessian=True)``).
# Wood (2011, sec. 3.5.1) prices exact second derivatives at ``O(M n p^2)`` and
# first derivatives at ``O(n p^2)``, and prefers the cheaper scheme only for
# large ``M``.  1e11 is about two seconds a REML iteration on the reference host
# (a freMTPL2 fit with one interaction pair is 2.7e11); below it a fit keeps the
# exact first-order Hessian and its established path.
_LEVERAGE_HESSIAN_MIN_GRAM_WORK = 1.0e11


def _usable_w_correction(correction, args: tuple, kwargs: dict, distribution):
    """``correction(*args, **kwargs)``; a ``LinAlgError`` out of it is no usable curvature here."""
    try:
        return correction(*args, **kwargs)
    except np.linalg.LinAlgError as error:
        raise ObservedModeNotConvergedError(
            f"REML W(rho) correction has no usable curvature at this point: {error}",
            hint=mode_certification_hint(distribution),
            # PIRLS reached a mode here; the correction built from it
            # is what failed, so the published wording must not assert
            # a PIRLS failure.
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

    Linear convergence at ``contraction`` a step reaches the score bar within
    the iterations left in the common case, and the search then finishes as it
    always has on the order-1 Hessian.
    """
    if step_reversals < 2 or not slow_reversal:
        return False
    steps = math.log(tol * score_scale / max(proj_grad_norm, 1e-300)) / math.log(contraction)
    return steps > iterations_left


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
    warm_lambdas: Mapping[str, float] | None = None,
    presolve: bool | None = None,
) -> REMLResult:
    """Optimize the direct REML objective via damped Newton (Wood 2011).

    ``warm_lambdas`` (a previous fit's estimates, keyed by component name)
    seeds the bootstrap and the first Newton iterate of every component it
    names; the rest bootstrap cold (``discrete.bootstrap_seed_lambdas``).

    A cold exact fit (no warm start) with at least ``_PRESOLVE_MIN_ESTIMATED``
    estimated smoothing parameters, Poisson with the log link, no
    structured-solver term and an iteration cap of at least ``_PRESOLVE_MIN_REML_ITER`` first
    runs the cached-W performance
    iteration (``optimize_discrete_reml_cached_w``) on this same lossless
    design until its objective settles, and the exact Newton loop starts
    from its smoothing parameters and coefficients: mgcv's ``outerPIsteps``
    (performance-iteration steps that initialize the outer iteration), taken
    to the cached-W fixed point.  The exact loop's gradient and stopping test
    are unchanged.  ``presolve`` overrides the rule for tests only.

    Two algorithm variants depending on ``discrete``:

    **Exact path** (``discrete=False``):
        W(rho)-corrected direct REML with gradient, Hessian, line search.

    **Discrete path** (``discrete=True``):
        Cached-W fREML optimizer (fewer data passes), delegated to
        ``optimize_discrete_reml_cached_w``.  This deliberately uses the
        BAM-style cached working/Fisher curvature approximation and bypasses
        ordinary observed-Hessian LAML for noncanonical links.
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
    w_correction_order = validate_w_correction_order(w_correction_order)
    if discrete:
        # The cached-W branch is the BAM-style performance iteration: it holds
        # W fixed within a step (no dW/drho term), so it does not reach the
        # exact route's optimum.  Its criterion is the binned rule's
        # (``classify_binned_reml_curvature``): Wood's observed-Hessian LAML for
        # Gamma/log, Tweedie/log, NB2/log, probit and cloglog, Fisher otherwise.
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
            max_analytical_per_w=max_analytical_per_w,
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
    # A model the structured solvers can also fit keeps the derivative Grams on
    # the dense solver too: both backends then build the same outer Hessian and
    # take the same REML path (tests/test_factor_smooth_structured_parity.py).
    structured_alternative = (
        select_structured_group(list(dm.group_matrices), groups, mode="auto").group_index
        is not None
    )
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
    # One-engine design §3.8: every PIRLS the criterion reads certifies its
    # mode at the bar this stopping tolerance needs (``mode_score``).
    mode_bar = mode_certification_bar(reml_tol)
    # The Laplace approximation's identified part (design §3.9,
    # ``reml.identified``): decided once, from the design and prior weights.
    identified = IdentifiedLaplace.for_design(
        dm, sample_weight, penalties, y=y, distribution=distribution, link=link
    )
    # A dense Fisher evaluation's identified part restricts the centred
    # Hessian its PIRLS decomposed; the cache holds it (no extra pass).
    identified_dense = bool(identified) and not use_structured
    # One-engine design §3.8: every PIRLS the criterion is evaluated at stops
    # on the certificate's own centred score over the identified coefficients.
    # A deviance change is blind to a coefficient whose rows carry a small
    # share of the likelihood, while its log-curvature enters log|H| at full
    # weight: the objective then moved with every warm start (stage-0
    # verifier census, raw_1e8 binomial).  A linearly constrained or SCOP
    # mode is certified by its inner QP/KKT state and keeps the deviance stop.
    pirls_convergence = (
        "mode_score"
        if use_observed_geometry
        or not any(
            getattr(group, "constraints", None) is not None
            or getattr(group, "monotone_engine", None) == "scop"
            for group in groups
        )
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
    # Only a rejected general raw-centering route crosses coefficient fits.
    # This owner dies with the optimizer; outside/finalization fits stay fresh.
    raw_moment_policy = TabmatCenteringState()
    fisher_data_reuse = _FisherDataReuse()
    # One flag for the fit's PIRLS solves: once a Gamma/log solve has had to
    # leave Fisher scoring for Newton (``fit_irls_direct``), the solves after
    # it start on Newton, since the same data and nearby smoothing parameters
    # give Fisher the same trouble.  A fit whose Fisher solves converge never
    # sets it.
    newton_latch = [False]

    # === Pre-solve: cached-W performance iteration on this same design ===
    # The cached-W loop (no dW/drho term) is cheap per iteration and covers
    # the far-from-optimum phase in which exact Newton spends most of its
    # iterations.  Its fixed point is a start, never a result: the exact loop
    # below evaluates, steps and stops on the exact criterion from there.
    presolve_beta: NDArray | None = None
    presolve_intercept: float | None = None
    presolve_centred: tuple[float, NDArray] | None = None
    if presolve is None:
        presolve = (
            not warm_lambdas
            and int(np.count_nonzero(estimated_mask)) >= _PRESOLVE_MIN_ESTIMATED
            and type(distribution) is Poisson
            and type(link) is LogLink
            and not use_observed_geometry
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

    # === Bootstrap: one FP step from conservative interaction penalties ===
    # Rich tensor interactions can explode under an almost-unpenalized
    # bootstrap fit. Keep main-effect bootstrap lambdas tiny, but start
    # interaction penalty components from a materially stronger seed. A warm
    # start bootstraps at its own lambdas instead (``bootstrap_seed_lambdas``).
    boot_lambdas = bootstrap_seed_lambdas(penalties, warm_lambdas, lambdas, estimated_mask)
    # After the pre-solve every estimated component starts warm, so the
    # bootstrap would fit the mode the first candidate fits again at the same
    # lambdas: the candidate starts from the pre-solve's mode instead.
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
    # Frozen directions from the previous iteration's active-set decision:
    # the stop criterion judges the ACTIVE set. An inferentially flat frozen
    # direction keeps a tiny persistent gradient forever (that is what makes
    # it flat), and counting it would spin the loop doing nothing until
    # max_reml_iter with every informative direction long determined.
    stop_criterion_frozen = None
    # The order-1 outer Hessian leaves out the d2W terms, so its Newton steps
    # converge only linearly where they omit real curvature (Nocedal and
    # Wright 2006, theorem 3.6): a step that keeps reversing while the
    # gradient shrinks by less than half a step is overshooting by a factor
    # the model gets wrong (measured: a Poisson log lambda alternating
    # +-0.02 with its gradient contracting 0.87 a step, 34 iterations to
    # 1e-9 against 15 with the exact Hessian). After two such steps in a row
    # in the endgame (the objective settled to the square root of the
    # tolerance, where a Newton step would square the error), each decreasing
    # the objective no more than its quadratic model predicted (a larger
    # decrease is not curvature but a moving mode, as a separated level
    # drifting to its bound), and when that rate would not reach the score
    # bar in the iterations left, the search takes the exact Hessian (Wood
    # 2011, full Newton) where the family and link supply it.
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
        # No interior mode at these lambdas: the penalized maximum holds rows
        # at the boundary of the family's mean space, where it is constrained,
        # not stationary, and the Laplace approximation (Wood, Pya & Saefken
        # 2016, JASA 111, section 3.1: a positive definite maximum of a
        # regular likelihood) does not exist.  Raising every estimated
        # penalty moves the mode towards the unpenalized fit, so the outer
        # iteration restores an interior start along that direction, in
        # doubling log-steps (the bracketing phase of a line search, Nocedal
        # & Wright 2006, section 3.5).  Each refit starts cold: a state at
        # the boundary is no start, since its rows' Fisher weights
        # mu / (1 - mu) pin them there and their working response
        # eta + (1 - mu) / mu is never inside.  A line-search trial without
        # an interior mode is rejected as any uncertified trial is.
        restoration_step = 1.0
        rho_at_boundary = rho_clipped
        while pirls_result.termination_reason == "mean_space_boundary":
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
        # What proves this mode stationary is PIRLS's own stop: the mode
        # certificate (``convergence="mode_score"``, one-engine design §3.8)
        # on every route that has one, the deviance stop for a linearly
        # constrained or SCOP mode.  A budget-exhausted candidate is therefore
        # not stationary on any path; the observed geometry's mode score below
        # is recorded as a diagnostic and never raises this flag.
        candidate_mode_stationary = bool(pirls_result.converged)
        if use_observed_geometry:
            if not pirls_result.converged and not stopped_on_iteration_budget(pirls_result):
                # Typed so a power search can score this point infeasible and
                # route around it, exactly as it does the certification
                # failure below. Every reason that still lands here names a
                # structural failure the mode score cannot speak to -- an
                # infeasible constrained mode, a missing inner-QP certificate,
                # a non-finite deviance. Budget exhaustion alone does not.
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
                # A budget-exhausted iterate now reaches this build, and an
                # iterate still mid-descent can carry observed information the
                # geometry refuses outright -- non-finite rows, a non-positive
                # intercept curvature. The line search already guards the same
                # call (it halves its step); here there is no step to halve, so
                # retype instead. A bare ValueError would escape every
                # `except ObservedModeNotCertifiedError` and kill a power
                # search that only needed to score this point infeasible.
                #
                # Only that one refusal is retyped. The build's other ValueErrors
                # report a violated caller contract -- a misshapen design, a bad
                # derivative_order, a penalty that is not PSD -- and catching
                # them here told the user PIRLS had reached no penalized mode at
                # this point, blaming the data's conditioning for a bug in the
                # call and inviting a power search to route quietly around it.
                # They now surface as the bugs they are.
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
                # An unusable score is the same kind of statement as the
                # unusable geometry just above: this iterate has no penalized
                # mode worth differentiating through. The answer is a shorter
                # step or a point scored infeasible, never a failed fit, and
                # there being no step to halve here it gets the same retype --
                # a bare ValueError would walk past every
                # `except ObservedModeNotCertifiedError` and kill a power search
                # that only needed to route around this point. The score's other
                # ValueErrors stay bare for the same reason the build's do: rows
                # that do not match the design or a penalty in the wrong
                # coordinates are bugs in the call, not conditioning.
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
            # The certificate is PIRLS's own stop residual at the mode it
            # returned (``convergence="mode_score"``, one-engine design §3.8):
            # ``candidate_mode_stationary`` above already reads it.  A mode
            # that misses the bar is never refused (owner decision 3,
            # 2026-09-30): it cannot authorise a convergence exit, so a fit
            # that never certifies ends not converged and says so.  The
            # geometry's own score is kept as a diagnostic.

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
                    "shoulder": [False] * m,
                    "proj_grad_signed": [0.0] * m,
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
        leverage_gradient = False
        leverage_hessian = False
        w_correction_args: tuple = ()
        w_correction_kwargs: dict[str, Any] = {}
        if not discrete:
            leverage_gradient = bool(
                type(distribution) is Binomial
                and type(link) is LogitLink
                and w_correction_order == 1
                and len(penalties) >= _LEVERAGE_GRADIENT_MIN_PENALTIES
                and geometry is None
                and isinstance(as_hessian_factor(reml_inverse), DenseHessianFactor)
            )
            leverage_hessian = bool(
                not leverage_gradient
                and not structured_alternative
                and w_correction_order == 1
                and len(penalties) >= _LEVERAGE_GRADIENT_MIN_PENALTIES
                and float(len(penalties)) * dm.n * dm.p * dm.p >= _LEVERAGE_HESSIAN_MIN_GRAM_WORK
            )
            # The correction is built from this iterate's weight derivatives, so
            # a LinAlgError out of it says the same thing the geometry seams do:
            # no usable curvature at this point. The only LinAlgError reachable
            # from here is a compact operator refusing its blocks -- naming the
            # reachable set rather than `w_derivatives`' current habits, because
            # it is the reachable set that makes this narrow enough not to
            # swallow a structural complaint. Structural refusals on this path
            # are ValueError/TypeError and still propagate bare.
            #
            # Deliberately NOT gated on `use_observed_geometry`. The correction
            # runs on the Fisher path too, and Tweedie + PowerLink is canonical
            # wherever power == 1 - p, so the curvature flips to "fisher" there
            # and no observed geometry is built. That point is not exotic and
            # not rare: `_search_brent` evaluates both `p_bounds` outright
            # before the optimiser runs, so PowerLink(-0.05) against the default
            # (1.05, 1.95) crosses on the FIRST evaluation; the grid methods
            # cross at any node equal to 1 - power; and a plain fit_reml at
            # Tweedie(p=1.5) + PowerLink(-0.5) -- the canonical parameterisation
            # -- crosses with no search at all. Gating here would make the same
            # failure routable at one power and fatal at its neighbour, and at a
            # bound would kill the search before it evaluated anything. The
            # message stays curvature-neutral for the same reason: it is
            # reached with `geometry is None`.
            w_correction_args = (dm, link, groups, pirls_result, reml_inverse, cand_lambdas)
            w_correction_kwargs = {
                "penalty_caches": penalty_caches,
                "sample_weight": sample_weight,
                "offset_arr": offset_arr,
                "distribution": distribution,
                "w_correction_order": w_correction_order,
                "reml_penalties": penalties,
                "geometry": geometry,
                "gradient_only": leverage_gradient,
                "leverage_hessian": leverage_hessian,
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
                elif len(w_corr) > 2:
                    # The leverage route's last entry: its cross terms, or None
                    # when the tables formed the gradient alone.
                    correction_mode = (
                        "leverage_hessian" if w_corr[-1] is not None else "leverage_tables"
                    )
                else:
                    correction_mode = "leverage_gradient"
            profile["reml_w_correction_mode"] = correction_mode
        _t_w_correction += _time.perf_counter() - _t0
        if w_corr is not None:
            _n_w_correction += 1
            grad_w_correction = w_corr[0]
            dH_extra = w_corr[1]
            # Second order: the second-derivative terms.  The leverage route:
            # the first-order cross terms its pass formed in place of
            # ``dH_extra``.  Either is an (m, m) term added to the Hessian.
            dH2_cross = w_corr[2] if len(w_corr) > 2 else None
            grad = grad_partial + grad_w_correction
        else:
            grad = grad_partial.copy()
            dH_extra = None
            dH2_cross = None
        # The leverage route read back from term-pair tables: an exact gradient
        # and no W(rho) cross terms (``reml_w_correction``).
        table_route = (
            leverage_hessian
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

        revalidated_hess = None
        # A loose reml_tol can be satisfied by an iteration whose starved
        # candidate PIRLS never certified: the compound criterion's
        # objective and gradient were then measured at a nonstationary
        # beta and prove nothing about the profile objective -- the same
        # evidence standard the line search and the all-frozen exit
        # enforce. Withholding the accept is benign: rho's step is tiny
        # here, so the following iterations are warm-started PIRLS
        # settles, and the exit fires once the mode certifies (measured:
        # one extra outer iteration on the starved fixture, landing
        # 7e-4 -> 8e-7 off the reference lambdas).
        mode_certified = trial_counts_as_precision_evidence(candidate_mode_stationary, obj)
        if candidate_convergence.converged and mode_certified:
            # The stop mask is iteration k-1's freeze decision intersected
            # with the current gradient arm -- which cannot see a
            # curvature-only re-activation: a masked direction can hold a
            # gradient below the 1e-7 freeze floor yet above the requested
            # reml_tol while its coupled partner's update raised its row
            # curvature past the bar. Before accepting, recompute the
            # freeze decision against the CURRENT Hessian; only masked
            # directions whose gradient exceeds the tolerance can change
            # the verdict, so the check is skipped when none do.
            masked_live = stop_criterion_frozen is not None and bool(
                np.any(
                    np.asarray(stop_criterion_frozen) & (np.abs(proj_grad) >= _tol * score_scale)
                )
            )
            # The compound test is first order.  A point it accepts must also
            # pass the second-order necessary condition to the precision asked
            # for (``concave_live_directions``): on a concave shoulder of a
            # plateau the gradient is small however far the optimum is.
            #
            # On the table route the Newton model leaves out the W(rho) cross
            # terms -(P + P')/2, P_ij = tr(H^-1 S_i H^-1 C_j), whose diagonal
            # -P_ii has no fixed sign.  The stop's second-order check and its
            # freeze decision read the v0.40.0 leverage route's Hessian
            # instead: one solved-row pass forms the cross terms here, at the
            # candidate stop only.  A stop they refuse steps on that Hessian
            # too, as v0.40.0 did: the step's own freeze then keeps the
            # directions the revalidation reactivated.  (Stepping on the
            # gradient-only Hessian refroze them, and the next stop was refused
            # again: on the 10-pair freMTPL2 fit two directions whose curvature
            # straddles the freeze bar, 0.0033 and 0.0030 without the cross
            # terms and 0.0038 and 0.0040 with them against 0.0035, held
            # gradients of 5e-4 and 1.7e-3 through ten refused stops.)
            stop_cross = None
            if table_route:
                _t0 = _time.perf_counter()
                stop_cross = _usable_w_correction(
                    reml_w_correction,
                    w_correction_args,
                    {**w_correction_kwargs, "lookup_tables": False},
                    distribution,
                )[2]
                _n_stop_cross_passes += 1
                _t_w_correction += _time.perf_counter() - _t0
            _t0 = _time.perf_counter()
            revalidated_hess = reml_direct_hessian(
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
            if stop_cross is not None:
                revalidated_hess = revalidated_hess + stop_cross
            concave_live = concave_live_directions(
                proj_grad,
                revalidated_hess,
                estimated_mask,
                objective=obj,
                tolerance=_tol,
                step_bound=max_newton_step,
            )
            if bool(np.any(concave_live)):
                # The score test passed but the second-order check did not:
                # record which shoulders refused the stop (score-test exit only).
                _n_concave_stop_refusals += 1
                if profile is not None:
                    live = np.flatnonzero(concave_live)
                    profile["reml_concave_stop_refusal"] = {
                        "iter": int(n_iter),
                        "names": [group_names[i] for i in live],
                        "gradient": [float(abs(proj_grad[i])) for i in live],
                        "curvature": [float(revalidated_hess[i, i]) for i in live],
                        "step_bound": float(max_newton_step),
                    }
            if not masked_live and not bool(np.any(concave_live)):
                _t_hessian += _time.perf_counter() - _t0
                converged = True
                termination_reason = "score_objective_tolerance"
                break
            revalidation = freeze_flat_directions(
                proj_grad,
                revalidated_hess,
                direction_penalty_ranks(penalties, penalty_ranks),
                estimated_mask,
                objective=obj,
                tolerance=_tol,
                step_bound=max_newton_step,
            )
            if profile is not None:
                profile["reml_freeze_revalidated"] = True
                # The revalidation IS the last freeze decision made: when
                # it authorizes the exit below, the published record must
                # describe it, not iteration k-1.
                profile["reml_freeze_decision"] = {
                    "names": list(group_names),
                    "proj_grad": [float(abs(v)) for v in proj_grad],
                    "hess_diag": [float(revalidated_hess[i, i]) for i in range(m)],
                    "row_curvature": [float(v) for v in revalidation.row_curvature],
                    "penalty_rank": [float(v) for v in revalidation.penalty_rank],
                    "normalized_curvature": [float(v) for v in revalidation.normalized_curvature],
                    "curvature_bar": float(revalidation.curvature_bar),
                    "score_scale": float(score_scale),
                    "estimated": [bool(v) for v in estimated_mask],
                    "frozen": [bool(v) for v in revalidation.frozen],
                    "shoulder": [bool(v) for v in np.asarray(revalidation.shoulder, dtype=bool)],
                    # The shoulder arm reads the sign (``concave_live_directions``).
                    "proj_grad_signed": [float(v) for v in proj_grad],
                }
            _t_hessian += _time.perf_counter() - _t0
            reactivated = concave_live | (
                ~np.asarray(revalidation.frozen)
                & (
                    np.asarray(stop_criterion_frozen)
                    if stop_criterion_frozen is not None
                    else np.zeros(m, dtype=bool)
                )
                & (np.abs(proj_grad) >= _tol * score_scale)
            )
            if not bool(np.any(reactivated)):
                converged = True
                termination_reason = "score_objective_tolerance"
                break
            # A re-activated direction with a live gradient: not converged.
            # Fall through into this iteration's Newton step, reusing the
            # Hessian just computed (with the stop's cross terms on the table
            # route).

        # Wood outer-Hessian update.  With ``w_correction_order=2`` this
        # includes the exact available second curvature derivatives; the
        # default order 1 is an exact objective/gradient with a modified
        # (quasi-Newton) Hessian.  Wood (2011) eq. 6.2 writes the
        # diagonal dS_i/dρ_i term as g_i + 0.5*r_i, where g_i is the fixed-W
        # gradient.  The W(rho) terms are differentiated separately through
        # dH_extra and dH2_cross; using the total gradient here would add the
        # first-order W correction twice on the diagonal.
        _t0 = _time.perf_counter()
        hess = (
            revalidated_hess
            if revalidated_hess is not None
            else reml_direct_hessian(
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
        )
        if use_structured and not (np.all(np.isfinite(hess)) and np.all(np.isfinite(grad))):
            # Prior weights at the edge of the float64 range can take the
            # structured factor's trace terms out of range (measured at a
            # uniform weight of 1e-200); no Newton step exists, so this is the
            # structured path's one clear error rather than eigh's.
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
                "estimated": [bool(v) for v in estimated_mask],
                "frozen": [bool(v) for v in frozen],
                "shoulder": [bool(v) for v in shoulder],
                # The shoulder arm reads the sign (``concave_live_directions``).
                "proj_grad_signed": [float(v) for v in proj_grad],
            }
            if shoulder.any():
                # Every iteration that steps a shoulder, with its signed
                # gradient and curvature: the last freeze decision alone
                # forgets a shoulder read earlier in the fit.
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
                # The Fisher path admits a PIRLS-exhausted candidate, so an
                # all-frozen verdict can be measured at a nonstationary
                # beta with the objective still moving. All frozen means
                # rho stops moving, so deferring is a warm-started PIRLS
                # settle at the same lambdas -- the stationary exit fires
                # only for a stationary mode with a stable objective
                # (measured: warm-start chaining makes the max_iter=1
                # all-null fixture identical to the reference by the time
                # this exit fires, so the gate defers nothing there).
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

        # Modified Newton: eigendecompose, flip negatives, floor small eigenvalues.
        # A concave shoulder takes its trust-region step instead: the model
        # decreases without bound along negative curvature, so the step lies
        # on the region's boundary (More & Sorensen 1979), here the step cap,
        # in the descent direction.  A flipped-curvature Newton step on an
        # exponential shoulder moves one e-fold per iteration.
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
        # The synchronous search owns fixed design coordinates and warm state.
        # Its rejected trials may share initial data, never their penalties.
        _initial_data_reuse = (
            _InitialDataReuse()
            if not use_observed_geometry and debug_recorder is None and trace_run is None
            else None
        )
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
                    _initial_data_reuse=_initial_data_reuse,
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
                # A trial the structured factor refuses (a tree pivot within
                # its certified uncertainty, a border block that is not
                # positive definite, ...) supplies no objective: keep the
                # retained state and try a shorter step towards it, as the
                # discrete driver does and as a trial whose observed geometry
                # is infeasible is handled below.  No other solver is tried;
                # a refused candidate or bootstrap fit still raises.
                _n_refused_structured_trials += 1
                step *= 0.5
                continue
            trial_mode_residual: float | None = None
            # Same standard as the candidate above: the certificate, where one
            # runs, is what establishes stationarity.
            trial_mode_stationary = bool(trial_result.converged)
            if use_observed_geometry:
                if not trial_result.converged and not stopped_on_iteration_budget(trial_result):
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
                    # A shorter step is the answer to a trial iterate the
                    # geometry refuses, so this rejection is a real line-search
                    # outcome and is counted as one. It is deliberately the only
                    # refusal handled here: a violated caller contract does not
                    # improve at half the step, and halving would retry the same
                    # malformed call until the line search ran out of trials and
                    # reported the bug as a conditioning failure.
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
                    # A score this trial iterate makes unusable says exactly
                    # what the refused geometry above says: there is no
                    # penalized mode here worth differentiating through. Both
                    # are answered by a shorter step, never by failing the fit,
                    # so this takes the same door and is counted as the
                    # line-search rejection it is. Nothing is recorded in
                    # `_rejected_trial_observed_mode_residual_max`: the score
                    # refused before producing a residual, exactly as the
                    # geometry refusal above has none to record. And as there,
                    # only that one refusal backtracks -- the score's other
                    # ValueErrors report rows that do not match the design or a
                    # penalty in the wrong coordinates, which no shorter step
                    # improves.
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
                # A refused trial supplies no objective or precision evidence.
                # Keep the retained state and try a shorter step. The current
                # candidate's objective and caller-contract errors still raise.
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

            if trial_accepted:
                if trial_mode_residual is not None:
                    _accepted_observed_mode_residual_max = max(
                        _accepted_observed_mode_residual_max,
                        trial_mode_residual,
                    )
                rho = rho_trial
                warm_beta = trial_result.beta.copy()
                warm_intercept = float(trial_result.intercept)
                warm_centred = centred_warm_start(trial_result)
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
                # a STATIONARY state: on the Fisher path an Armijo-accepted
                # trial that exhausted max_pirls_iter is nonstationary, and
                # reusing it would put the next gradient/Hessian at the wrong
                # coefficients instead of warm-start refitting them.  Stationary
                # means PIRLS's own stop (``trial_mode_stationary`` above), on
                # every path: the geometry's mode score is a diagnostic.
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

        _initial_data_reuse = None
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
            # above the bar and stays an honest failure. The same holds in
            # objective units: when the Newton model of the active set
            # predicts a decrease below the stop resolution (half the
            # squared decrement of the uncapped step just tried, its
            # Hessian unmodified), the optimum is resolved with the gradient
            # above its bar but within a decade of it -- #459 measured that
            # decrease at 1.2e-8 against evaluation noise of 4.2e-7, so
            # rounding alone decided whether the full step was accepted.
            active_grad_norm = (
                float(np.max(np.abs(np.where(frozen, 0.0, proj_grad)))) if proj_grad.size else 0.0
            )
            dead_evidence = evaluated_feasible_trial and trial_counts_as_precision_evidence(
                candidate_mode_stationary, obj
            )
            dead_predicted_decrease = (
                None
                if max_delta > max_newton_step or bool(shoulder.any())
                else newton_predicted_decrease(
                    grad_sub, eigvals_h, eigvecs_h, eigenvalue_floor=eig_floor
                )
            )
            termination_reason = classify_dead_feasible_exit(
                active_grad_norm,
                objective=obj,
                tolerance=_tol,
                evaluated_trial=dead_evidence,
                predicted_decrease=dead_predicted_decrease,
            )
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
                }
            # The second-order check (``concave_live_directions``) gates the
            # score-test exit only.  A shoulder here is flat by construction,
            # its gradient under the precision floor this exit certifies, and
            # over the shortest step just rejected (5/128 in log lambda) its
            # model promises at most 0.041 times that floor: about four times
            # the objective's resolution at the default reml_tol (1e-9), and
            # 4e-9 / reml_tol times it at a tighter one.  A decrease the
            # rejected trials have just failed to show is no ground to refuse
            # a certified point.  (The native-Windows flip of
            # test_sz_aliased_levels_converge_under_reml[one_row] first blamed
            # on refusing here was the unsigned shoulder read the signed rule
            # removes.)  test_a_dead_search_beside_a_live_shoulder_certifies
            # pins this exit beside a shoulder on both engines.
            converged = termination_reason == "converged_at_precision"
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
