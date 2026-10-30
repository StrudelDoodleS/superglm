"""The direct IRLS fit: ``fit_irls_direct``, its method object ``_IRLSDirectFit`` and their helpers.

The package's ``__init__`` describes the solver and re-exports the names other modules use.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, replace
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm import _held_warnings as held_warnings
from superglm._fit_trace import TraceRun
from superglm._group_matrix._group_matrix_tabmat import (
    _defer_raw_spline_tabmat_plan,
    _is_raw_spline_tabmat_centering_candidate,
)
from superglm._utils import weighted_row_sum
from superglm.distributions import (
    Binomial,
    Distribution,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
)
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import IdentityLink, Link, LogitLink, LogLink, stabilize_eta
from superglm.solvers.centered_system import (
    CenteredSystem,
    TabmatCenteringState,
    _FisherDataReuse,
    _InitialDataReuse,
    build_centered_system,
    grouped_augmented_factor,
    grouped_augmented_factor_rhs,
    refresh_centered_rhs,
)
from superglm.solvers.constrained_qp import _feasibility_slack
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom
from superglm.solvers.irls_direct._convergence import _IRLSDirectConvergence
from superglm.solvers.irls_direct._dense import _IRLSDirectDenseStep
from superglm.solvers.irls_direct._finalise import _IRLSDirectFinalGeometry
from superglm.solvers.irls_direct._mean_space import _IRLSDirectMeanSpaceOverlay
from superglm.solvers.irls_direct._qp import _QP_FEASIBILITY_TOL, _IRLSDirectQPStep
from superglm.solvers.irls_direct._qr import _IRLSDirectQRStep
from superglm.solvers.irls_direct._recorder import _IRLSDirectRecorder
from superglm.solvers.irls_direct._scop import (
    SCOPRunCentring,
    _carry_refusals,
    _IRLSDirectSCOPStep,
    _seed_refusals,
)
from superglm.solvers.irls_direct._search import _IRLSDirectTrialSearch
from superglm.solvers.irls_direct._step import _IRLSDirectStep
from superglm.solvers.irls_direct._structured import _IRLSDirectStructuredStep
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _irls_objective_scale,
    _IRLSState,
    interior_start_intercept,
    link_domain_violation,
    mean_space_violation,
)
from superglm.solvers.mode_score import (
    MODE_CERTIFICATION_BAR,
    MODE_RESOLVE_CAP,
    ModeResidual,
    centred_intercept_remainder,
    centred_matvec,
    penalized_mode_residual,
    prior_weighted_centre,
)
from superglm.solvers.pirls import (
    PIRLSResult,
    TerminationReason,
    _positive_working_weight_stats,
)
from superglm.solvers.rank import (
    RankDecomposition,
    RankInfo,
    decompose_factor,
    decompose_gram_if_authoritative,
    decompose_symmetric,
)
from superglm.solvers.structured import resolve_structured_backend
from superglm.solvers.working_rows import (
    CoefficientWorkingRows,
    coefficient_initial_intercept,
    coefficient_working_rows,
    concave_observed_rows,
    fisher_working_weights,
    pearson_chi2,
    supports_observed_newton,
)
from superglm.types import GroupSlice, LinearConstraintSet, PenaltyComponent

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)


@dataclass(frozen=True)
class _CenteredFactorCertification:
    """One fit-local factor certificate tied to immutable centered geometry."""

    system: CenteredSystem
    factor: NDArray
    decomposition: RankDecomposition
    transformed_rhs: NDArray | None


def _has_constant_irls_weights(family: Distribution, link: Link) -> bool:
    """Return True when PIRLS weights are independent of ``mu``.

    The direct solver can reuse X'WX only when
    ``(dmu/deta)^2 / V(mu)`` is exactly constant.  Keep this deliberately
    conservative so performance never changes the fitted problem.
    """
    from superglm.distributions import Gamma, Gaussian, Poisson
    from superglm.links import IdentityLink, LogLink, SqrtLink

    return (
        (type(family) is Gaussian and type(link) is IdentityLink)
        or (type(family) is Gamma and type(link) is LogLink)
        or (type(family) is Poisson and type(link) is SqrtLink)
    )


def _separation_weight_ratio(
    curvature_source: str,
    w_ratio: float,
    *,
    family,
    link,
    mu: NDArray,
    eta: NDArray,
    weights: NDArray,
) -> float:
    """The working-weight ratio the separation bar was set on: expected curvature.

    Observed rows scale each Fisher weight by alpha = 1 + (y - mu)(V'/V + g''/g'),
    (2 - p) + (p - 1) y / mu for Tweedie/log, whose spread is not separation.
    """
    if curvature_source == "fisher":
        return w_ratio
    fisher = fisher_working_weights(
        distribution=family, link=link, mu=mu, eta=eta, sample_weight=weights
    )
    return _positive_working_weight_stats(fisher)[2]


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


def _safe_decompose_H(H: NDArray, residual_tol: float = 1e-6) -> tuple[NDArray, float, bool]:
    """Compatibility wrapper returning inverse, pseudo-logdet, and fast-path flag."""
    decomposition = decompose_symmetric(H, residual_tol=residual_tol)
    cholesky_ok = decomposition.method in {"cholesky", "pivoted_cholesky"}
    return decomposition.pseudo_inverse(), decomposition.log_pdet, cholesky_ok


def _refuse_scop_with_linear_constraints(groups: list[GroupSlice]) -> None:
    """Refuse a fit with both shape-constrained (SCOP) and linearly constrained groups.

    The SCOP step takes one Newton step on every coefficient and has no
    constrained QP, so a linear constraint beside it would be checked at the
    end but never enforced by a step.  Every fit entry point refuses the mix;
    the solver refuses it as well, so each fit has exactly one step method.
    """
    shaped = [group.name for group in groups if group.monotone_engine == "scop"]
    bounded = [group.name for group in groups if group.constraints is not None]
    if shaped and bounded:
        raise NotImplementedError(
            "fit_irls_direct cannot fit shape-constrained (SCOP) groups and linearly "
            f"constrained groups together: {', '.join(map(repr, shaped))} "
            f"{'is' if len(shaped) == 1 else 'are'} shape-constrained and "
            f"{', '.join(map(repr, bounded))} {'has' if len(bounded) == 1 else 'have'} "
            "linear constraints. Use one constraint method for every constrained term "
            "(all of them PSplines, which use SCOP, or none of them), or drop one of the "
            "constraints."
        )


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
    scop_state_init: dict[int, dict] | None = None,
    debug_recorder=None,
    debug_context: dict[str, object] | None = None,
    compute_rank_info: bool = True,
    _return_working_system: bool = False,
    _compute_fit_statistics: bool = True,
    _compute_reml_geometry: bool = True,
    _retain_reml_decomposition: bool = False,
    _use_observed_newton: bool = True,
    _working_newton: bool = False,
    _newton_latch: list[bool] | None = None,
    _deviance_init: float | None = None,
    trace_run: TraceRun | None = None,
    trace_purpose: str = "fit",
    _compute_scop_postfit_inference: bool = True,
    separation: str = "warn",
    *,
    weight_semantics: str,
    _initial_data_reuse: _InitialDataReuse | None = None,
    _raw_moment_policy: TabmatCenteringState | None = None,
    _fisher_data_reuse: _FisherDataReuse | None = None,
    _scop_run_centring: SCOPRunCentring | None = None,
    _laplace_excluded: tuple[int, ...] = (),
    _mode_bar: float | None = None,
    _compensate_centred_intercept: bool = True,
    _centred_init: tuple[float, NDArray] | None = None,
    _defer_rank_info: list[Callable[[], RankInfo]] | None = None,
) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
    """Fit by direct IRLS (``_IRLSDirectFit``, whose docstring documents the parameters).

    A structured factor that cannot proceed raises ``StructuredSolverError``
    under every ``direct_solve`` (one-engine design §6, decision 6): no fit is
    rerun on another solver.  A Fisher-data reuse cache is cleared when the
    fit fails or ends unconverged.  Shape-constrained (SCOP) groups and
    linearly constrained groups are refused together
    (``_refuse_scop_with_linear_constraints``).
    """
    result = None
    try:
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        _refuse_scop_with_linear_constraints(groups)
        result = _fit_irls_direct_once(
            X=X,
            y=y,
            weights=weights,
            family=family,
            link=link,
            groups=groups,
            lambda2=lambda2,
            offset=offset,
            beta_init=beta_init,
            intercept_init=intercept_init,
            max_iter=max_iter,
            tol=tol,
            return_xtwx=return_xtwx,
            profile=profile,
            cache_out=cache_out,
            record_diagnostics=record_diagnostics,
            direct_solve=direct_solve,
            convergence=convergence,
            S_override=S_override,
            reml_penalties=reml_penalties,
            return_scop_state=return_scop_state,
            scop_state_init=scop_state_init,
            debug_recorder=debug_recorder,
            debug_context=debug_context,
            compute_rank_info=compute_rank_info,
            _return_working_system=_return_working_system,
            _compute_fit_statistics=_compute_fit_statistics,
            _compute_reml_geometry=_compute_reml_geometry,
            _retain_reml_decomposition=_retain_reml_decomposition,
            _use_observed_newton=_use_observed_newton,
            _working_newton=_working_newton,
            _newton_latch=_newton_latch,
            _deviance_init=_deviance_init,
            trace_run=trace_run,
            trace_purpose=trace_purpose,
            _compute_scop_postfit_inference=_compute_scop_postfit_inference,
            separation=separation,
            weight_semantics=weight_semantics,
            _initial_data_reuse=_initial_data_reuse,
            _raw_moment_policy=_raw_moment_policy,
            _fisher_data_reuse=_fisher_data_reuse,
            _scop_run_centring=_scop_run_centring,
            _laplace_excluded=_laplace_excluded,
            _mode_bar=_mode_bar,
            _compensate_centred_intercept=_compensate_centred_intercept,
            _centred_init=_centred_init,
            _defer_rank_info=_defer_rank_info,
        )
        return result
    finally:
        if _fisher_data_reuse is not None and (result is None or not result[0].converged):
            _fisher_data_reuse.clear()


def _certify_centered_factor(
    dm: DesignMatrix,
    _fisher_data_reuse: _FisherDataReuse | None,
    cache: list[_CenteredFactorCertification | None],
    system: CenteredSystem,
    W_current: NDArray,
    *,
    response: NDArray | None = None,
) -> _CenteredFactorCertification:
    """Return a factor certificate for one immutable centered geometry."""
    if _fisher_data_reuse is not None:
        _fisher_data_reuse.clear()
    cached = cache[0]
    same_geometry = bool(
        cached is not None
        and cached.system.data_gram is system.data_gram
        and cached.system.penalty is system.penalty
        and cached.system.hessian is system.hessian
        and cached.system.mean_x is system.mean_x
        and cached.system.sum_w == system.sum_w
    )
    # A refreshed RHS can share the exact weighted Gram while differing in
    # a factor-resolvable direction that normal equations round away. The
    # immutable CenteredSystem instance identifies one RHS generation, so
    # transformed RHS reuse needs only identity—not an O(n) response copy
    # and comparison. Geometry-only terminal consumers may reuse the same
    # compact factor across refreshed constant-weight systems.
    if (
        same_geometry
        and cached is not None
        and (response is None or (cached.system is system and cached.transformed_rhs is not None))
    ):
        return cached

    centre, centre_lo = system.centre_pair()
    if response is None:
        factor = grouped_augmented_factor(
            dm,
            W_current,
            system.penalty,
            center=centre,
            center_lo=centre_lo,
        )
        transformed_rhs = None
    else:
        factor, transformed_rhs = grouped_augmented_factor_rhs(
            dm,
            W_current,
            system.penalty,
            response=response,
            center=centre,
            center_lo=centre_lo,
        )
    factor_decomposition = decompose_factor(
        factor,
        retain_factor_solve=transformed_rhs is not None,
    )
    cached = _CenteredFactorCertification(
        system=system,
        factor=factor,
        decomposition=factor_decomposition,
        transformed_rhs=transformed_rhs,
    )
    cache[0] = cached
    return cached


def _authoritative_gram_rank(
    memo: list[tuple[NDArray, RankDecomposition | None] | None], matrix: NDArray
) -> RankDecomposition | None:
    """``decompose_gram_if_authoritative(matrix)``, taken once per immutable matrix.

    The last Gram decomposition this fit took, with the matrix it was taken
    of.  ``decompose_gram_if_authoritative`` under the default policy is a
    pure function of the matrix's values, and a read-only array that owns its
    data (every ``CenteredSystem`` Hessian, ``centered_system._freeze``) is
    never reopened: nothing in superglm calls ``setflags(write=True)`` on a
    Hessian it holds, though NumPy would allow it.  That is the invalidation
    rule this memo rests on, so the same object has the same decomposition
    and the same authority verdict.  The dense step's iteration and the centred
    finaliser's REML slope read it: a finaliser that takes the system its
    last step solved (``_return_working_system``), or a Hessian object the
    fit kept across iterations, reads that step's decomposition.  Owner and
    lifetime: this solve (``memo`` is the fit's one-entry list); it holds
    one matrix, replaced by the next decomposition of another.  A matrix
    that is writeable or a view is never held.
    """
    held = memo[0]
    if held is not None and held[0] is matrix:
        return held[1]
    decomposition = decompose_gram_if_authoritative(matrix)
    if not matrix.flags.writeable and matrix.flags.owndata:
        memo[0] = (matrix, decomposition)
    return decomposition


def _fit_irls_direct_once(**kwargs):
    """Run one ``_IRLSDirectFit``: the module-level name callers and tests wrap."""
    return _IRLSDirectFit(**kwargs).run()


class _IRLSDirectCurvaturePolicy:
    """Fisher scoring or the observed information: the curvature of the fit's working rows.

    Decided once, at setup, from the family, the link and the route
    (one-engine design §3.11), and switched at most once, by the stop rule:

    - ``observed``: the working rows carry the observed information (Newton
      steps), not Fisher's.  Every line search of an observed fit rejects a
      trial whose observed rows are not finite (``observed_rows_invalid``).
    - ``stagnation_newton_available``: a declared Newton pair kept on Fisher
      rows for its constant weights (Gamma/log), which switches to Newton if
      its Fisher score stagnates (``switch_to_newton``), and starts on Newton
      when an earlier solve of the same REML fit switched (the latch).
    - ``weighted_gram_reuse``: the working weights are the same at every
      iterate (constant Fisher weights on a route whose centred system caches
      them), so the fit's centred system may reuse its weighted Gram and the
      caller's Fisher data.  The switch turns it off.

    Each is read where it is used, never cached past the switch.  The policy
    forms the working rows in its curvature (``working_rows``) and keeps the
    family, link, response and prior weights they read, and no reference to
    the fit.
    """

    __slots__ = (
        "observed stagnation_newton_available weighted_gram_reuse _latch _family _link _y _weights"
    ).split()

    def __init__(
        self,
        family: Distribution,
        link: Link,
        y: NDArray,
        weights: NDArray,
        *,
        use_observed_newton: bool,
        working_newton: bool,
        return_working_system: bool,
        latch: list[bool] | None,
        has_constraints: bool,
        has_scop: bool,
        use_structured: bool,
    ) -> None:
        # Laplace-approximate REML is built on the observed Hessian, so its PIRLS
        # runs on full-Newton weights (Wood 2011, JRSSB 73(1), section 3; Wood, Pya
        # & Saefken 2016, JASA, section 3.3): Fisher scoring shares the mode but
        # converges only linearly under a non-canonical link. Newton starts only
        # where the Fisher weights vary, so no constant-weight Gram or Fisher-data
        # cache exists for it to invalidate. Gamma/log keeps Fisher: its constant
        # Fisher weights reuse one weighted Gram that Newton would rebuild every
        # iteration; the mode certificate's bar is set by the REML tolerance
        # (``mode_score.mode_certification_bar``), which Fisher's linear rate
        # reaches wherever Fisher scoring contracts.  It need not: Fisher's error
        # map ``I - F^-1 H`` has an eigenvalue below -1 where the observed
        # information ``w y / mu`` exceeds twice the Fisher information ``w``
        # along a direction, as heavy-tailed severities give, and there the score
        # stagnates above the bar; nearer 0 Fisher contracts too slowly to reach
        # it in the budget.  Such a solve switches to Newton once its resolved
        # score shows it (``mode_score.rate_exhausts_budget``; Green 1984, JRSSB
        # 46, section 2.3), and keeps it to the end; a solve whose Fisher score
        # reaches the bar in time never switches.
        # The exported geometry is Fisher either way (``export_rows``), and the
        # weighted-Gram cache is off while Newton runs.  Otherwise the curvature
        # is decided here, once, from the family, the link and the route
        # (one-engine design §3.11).
        # An observed iterate whose Hessian the structured factor refuses as not
        # positive definite takes a Levenberg shift inside the same curvature
        # (``_levenberg_shifted_operator``), and a trial whose observed rows are
        # not finite is rejected by the line search.
        # The fREML performance iteration (``_return_working_system``) takes
        # Newton rows when its optimizer asks (``_working_newton``): the discrete
        # route's working model then carries the observed curvature its REML
        # criterion is (``reml.discrete``).
        # A shape-constrained (SCOP) fit takes the same rows: its joint Newton
        # step is Pya & Wood's (2015, Appendix 1) full Newton on the observed
        # information, and the REML geometry it hands on reads Fisher rows at the
        # retained model either way.
        constant_weights = _has_constant_irls_weights(family, link)
        observed = bool(
            use_observed_newton
            and not has_constraints
            and (
                (working_newton and concave_observed_rows(family, link))
                if return_working_system
                else supports_observed_newton(family, link) and not constant_weights
            )
        )
        stagnation_newton_available = bool(
            use_observed_newton
            and not has_constraints
            and not has_scop
            and not return_working_system
            and supports_observed_newton(family, link)
            and not observed
        )
        if stagnation_newton_available and latch is not None and latch[0]:
            observed = True
        self.observed = observed
        self.stagnation_newton_available = stagnation_newton_available
        # Observed rows are not the constant Fisher weights this reuse assumes.
        self.weighted_gram_reuse = (
            constant_weights and not has_scop and not use_structured and not observed
        )
        self._latch = latch
        self._family, self._link, self._y, self._weights = family, link, y, weights

    def working_rows(self, mu: NDArray, eta: NDArray) -> CoefficientWorkingRows:
        """The iterate's working rows in the current curvature (``coefficient_working_rows``)."""
        return coefficient_working_rows(
            distribution=self._family,
            link=self._link,
            y=self._y,
            mu=mu,
            eta=eta,
            sample_weight=self._weights,
            prefer_observed=self.observed,
        )

    def observed_rows_invalid(self, candidate: _IRLSState) -> bool:
        """Whether a trial's observed rows are not finite, which rejects it.

        One-engine design §3.11: finiteness is the one condition the declared
        observed rows keep; a trial without finite rows is not a state the
        next Newton step can start from. Every line search of an observed fit
        reads it, the SCOP one included.
        """
        rows = coefficient_working_rows(
            distribution=self._family,
            link=self._link,
            y=self._y,
            mu=candidate.mu,
            eta=candidate.eta,
            sample_weight=self._weights,
            prefer_observed=True,
        )
        return rows.rejection_reason is not None

    def switch_to_newton(self, recorder: _IRLSDirectRecorder) -> None:
        """Newton steps on the observed information for the iterations left.

        Fisher scoring near the mode contracts too slowly to meet the bar in
        the budget, or not at all: the observed information is positive for
        Gamma/log (``w y / mu``), so the penalized Hessian stays positive
        definite.  The weights now change at every iterate, so the weighted
        Gram is no longer reused, and the REML fit's later solves start on
        Newton (the latch).
        """
        self.observed = True
        self.weighted_gram_reuse = False
        if self._latch is not None:
            self._latch[0] = True
        recorder.count("irls_fisher_newton_switches")


class _IRLSDirectFitFields:
    """The fit's state beyond its parameters, by the method that first sets it.

    Slots, not an instance dict: a field is read and written at a fixed offset, not
    hashed into a dict of some two hundred entries, and a misspelt field is an error.
    """

    __slots__ = (
        # run
        "recorder"
        # _setup_design
        " dm n p gms objective_merit_scale structured_decision _use_structured step"
        " _direct_fallback_reason _state_center beta intercept _start_lowered S"
        " _penalty_curvature"
        # _setup_routes
        " qp _scop_run _has_scop curvature"
        # _setup_systems
        " mean_space _constant_centered_cache _constant_centered_z"
        " certify_centered_factor authoritative_gram_rank _initial_data_key _tabmat_split"
        " _tabmat_centering_state"
        " fixed_owners _initial_data_pending"
        # _setup_loop
        " converged _last_working_centered _last_working_offset_mean _score_centre"
        " stop_rule trial_search"
        " eta_unclipped eta mu max_halving _consecutive_svd"
        " termination_reason mode_bar committed"
        # _working
        " rank_truncated used_rank_certification proposal_centred_intercept"
        " working_eta_unclipped working_eta working_mu working_rows W z newton_score"
        " w_ratio"
        # the step (``step.propose``)
        " _cond_est _used_svd"
        # the line search (``trial_search``, or the step's own)
        " step_rejected dev n_halvings decision proposal_state retained"
        # the stop rule (``stop_rule``) and the mean space's judge
        " convergence_label convergence_tolerance dev_rel_change coef_change"
        " convergence_value converged_this_iter score_stagnated"
        # run's loop, after the verdict
        " deviance_relative_change"
        # _finish
        " centred_intercept_lo"
        # _terminal_rows
        " export_rows"
    ).split()


@dataclass(eq=False, repr=False, slots=True)
class _IRLSDirectFit(_IRLSDirectFitFields):
    """Fit a penalised GLM via direct IRLS (no BCD).

    Solves β = (X'WX + S)⁻¹ X'Wz at each iteration.  Uses gram-based
    operations to form X'WX without materialising the full (n, p) dense
    matrix.  For discretized groups (DiscretizedSSPGroupMatrix), this
    reduces the per-iteration cost from O(n·p²) to O(n_bins·K²).

    Returns (PIRLSResult, XtWX_S_inv) where XtWX_S_inv is the (p, p)
    profiled-intercept slope inverse from the final iteration, reusable for
    REML trace terms.

    Parameters
    ----------
    X : DesignMatrix or ndarray
        Design matrix (per-group or dense).
    y : (n,) array
        Response variable.
    weights : (n,) array
        Fitting weights.  ``weight_semantics`` says what they mean, and
        decides only the likelihood size the published dispersion divides by;
        the working weights themselves are the same under either reading.
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
        Deviance convergence tolerance (default 1e-8).
    return_xtwx : bool
        If True, also return the final weighted Gram matrix X'WX. Used by the
        REML outer loop to avoid rebuilding X'WX in cheap iterations when W is
        held fixed.
    compute_rank_info : bool
        If False, skip data-subspace metadata used only by retained-fit
        inference. Intermediate REML fits still compute the coefficient and
        augmented decompositions needed for their objective and EDF.
    _defer_rank_info : list, optional
        With ``compute_rank_info=False``, a list to which a dense fit with
        statistics appends a zero-argument callable returning the
        ``RankInfo`` it skipped, bit-for-bit what ``compute_rank_info=True``
        attaches. For a caller that learns only after the fit whether it is
        the one published (NB2 auto-theta REML refits).
    _return_working_system : bool
        Internal fREML performance-iteration mode. Return the centered system
        used for the last coefficient update instead of rebuilding it at the
        proposed coefficients. The authoritative final fit must leave this
        False so exported inference remains tied to the retained model.
    _compute_fit_statistics : bool
        Internal optimization switch. If False, omit EDF and scale summaries
        that the fREML outer loop does not consume. The authoritative final
        fit must leave this True.
    _compensate_centred_intercept : bool
        Internal switch for in-loop REML fits that compute statistics (the
        bootstrap, and traced candidates and trials): False keeps their eta
        the solver's ``alpha + X~ beta`` bit for bit.  With it and
        ``_compute_fit_statistics`` True, a centred Gaussian identity fit
        publishes its intercept as the compensated pair ``(alpha, alpha_lo)``
        (``mode_score.centred_intercept_remainder``) and its deviance and
        scale at that predictor.
    _centred_init : (float, ndarray), optional
        The warm start's centred state ``(alpha, c)``: its predictor is
        ``alpha + (X - 1 c') beta_init + offset`` (``PIRLSResult.centred_intercept``
        and ``state_center``; ``centred_warm_start``).  The fit carries it to
        its own centre exactly instead of taking it back from
        ``intercept_init``, whose rounding cancels ``c' beta`` at a column's
        offset.  Ignored without ``beta_init`` (the state's own ``beta``) or
        when ``intercept_init`` is not the start.
    _compute_reml_geometry : bool
        Internal SCOP-candidate switch. If False, omit the generic profiled
        slope inverse, determinant, and rank because the caller replaces them
        with one joint latent-coordinate LAML geometry. This requires both
        retained rank metadata and fit statistics to be disabled. Public and
        terminal fits must leave this True.
    _newton_latch : list of one bool, optional
        Internal, shared by the PIRLS solves of one REML fit.  A Gamma/log
        solve that switched from Fisher to Newton sets it, and a later solve
        of the fit that finds it set starts on Newton steps.  ``None`` (the
        default) shares nothing.
    _working_newton : bool
        Internal, read only with ``_return_working_system``: the fREML
        performance iteration's update takes observed (Newton) rows instead
        of Fisher's, for a pair whose observed rows are never negative
        (``working_rows.concave_observed_rows``).  The discrete optimizer
        asks for it when the pair's binned REML curvature is observed
        (``classify_binned_reml_curvature``), so its working model is the
        criterion it publishes.
    _use_observed_newton : bool
        Internal curvature switch. When enabled, a fit whose observed
        information is supported and whose Fisher weights vary (Tweedie/log)
        takes exact observed-Newton steps on every iteration, a
        shape-constrained (SCOP) fit's joint Newton step included; a Gamma/log
        fit, whose Fisher weights are constant, starts on Fisher scoring and,
        without SCOP groups, switches to Newton only when its resolved score
        is measured to stall (``_IRLSDirectCurvaturePolicy``) or when
        ``_newton_latch`` is set; no rejected step switches the curvature
        (one-engine design §3.11). As to this switch, unsupported pairs and
        constrained routes retain Fisher scoring, and the fREML performance
        iteration (``_return_working_system``) takes Newton rows only when
        ``_working_newton`` asks.  The binomial/log mean space's Newton steps
        are separate: they run on the dense, QP and structured routes once its
        certificate engages them (``_IRLSDirectMeanSpaceOverlay``).  Public
        callers should leave this True.
    _deviance_init : float, optional
        Previously evaluated deviance at ``beta_init``/``intercept_init``.
        Used by private fREML steps to avoid repeating a full response scan.
    _compute_scop_postfit_inference : bool
        Internal SCOP EFS switch. Candidate modes leave this False; the
        terminal/public mode installs covariance and EDF exactly once.
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
        Slope block of the full augmented Hessian inverse after profiling the
        intercept: ``(X_c' W X_c + S)^+``.
    """

    X: NDArray | DesignMatrix
    y: NDArray
    weights: NDArray
    family: Distribution
    link: Link
    groups: list[GroupSlice]
    lambda2: float | dict[str, float]
    offset: NDArray | None = None
    beta_init: NDArray | None = None
    intercept_init: float | None = None
    max_iter: int = 100
    tol: float = 1e-8
    return_xtwx: bool = False
    profile: dict | None = None
    cache_out: dict | None = None
    record_diagnostics: bool = False
    direct_solve: str = "auto"
    convergence: str = "deviance"
    S_override: NDArray | None = None
    reml_penalties: list[PenaltyComponent] | None = None
    return_scop_state: bool = False
    scop_state_init: dict[int, dict] | None = None
    debug_recorder: Any = None
    debug_context: dict[str, object] | None = None
    compute_rank_info: bool = True
    _return_working_system: bool = False
    _compute_fit_statistics: bool = True
    _compute_reml_geometry: bool = True
    _retain_reml_decomposition: bool = False
    _use_observed_newton: bool = True
    _working_newton: bool = False
    _newton_latch: list[bool] | None = None
    _deviance_init: float | None = None
    trace_run: TraceRun | None = None
    trace_purpose: str = "fit"
    _compute_scop_postfit_inference: bool = True
    separation: str = "warn"
    _: KW_ONLY
    weight_semantics: str
    _initial_data_reuse: _InitialDataReuse | None = None
    _raw_moment_policy: TabmatCenteringState | None = None
    _fisher_data_reuse: _FisherDataReuse | None = None
    _scop_run_centring: SCOPRunCentring | None = None
    _laplace_excluded: tuple[int, ...] = ()
    _mode_bar: float | None = None
    _compensate_centred_intercept: bool = True
    _centred_init: tuple[float, NDArray] | None = None
    _defer_rank_info: list[Callable[[], RankInfo]] | None = None

    def run(self) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
        self._setup_design()
        self.recorder = recorder = _IRLSDirectRecorder(self)
        self._setup_routes()
        self._setup_systems()
        recorder.start()
        self._setup_loop()
        step, stop_rule = self.step, self.stop_rule
        # the step's own line search, or the shared one
        search = self.trial_search.search if step.search is None else step.search
        for it in range(self.max_iter):
            self._working(it)
            if step.propose(self, it):
                break
            search(self, it)
            stop_rule.judge(self, it)
            recorder.iteration(self, it)
            # Taken from the state the iteration started at, before the retained
            # state is committed, because the post-loop separation backstop needs
            # the final iteration's movement: read after the loop it would compare
            # the retained state with itself, identically zero under budget
            # exhaustion, and any stagnation test on it would be vacuous.
            dev, dev_prev = self.dev, self.committed.deviance
            deviance_relative_change = abs(dev - dev_prev) / (abs(dev_prev) + 1)
            self.deviance_relative_change = deviance_relative_change
            logger.info(
                "  irls_direct iter=%3d  dev=%12.1f  delta=%10.2e",
                it + 1,
                dev,
                deviance_relative_change,
            )

            if not np.isfinite(dev):
                logger.warning(f"IRLS direct non-finite deviance at iter={it + 1}: dev={dev:.2e}")
                break
            if recorder.record_debug_rows:
                recorder.debug_row(self, it)

            if self.step_rejected:
                logger.warning(
                    "IRLS direct rejected all trial steps at iter=%d; restored committed state",
                    it + 1,
                )
                break

            self.committed = self.retained
            if self.converged_this_iter:
                self.converged = True
                break
            if self.score_stagnated:
                break
        recorder.stop(it)
        self._finish(it)
        self._terminal_rows()
        recorder.profile_fit(self, it)
        # the step's finaliser: the retained model's geometry
        return self._publish(it, step.finalise(self))

    def _setup_design(self) -> None:
        X, family, link, groups, weights = self.X, self.family, self.link, self.groups, self.weights
        offset, S_override, intercept_init = self.offset, self.S_override, self.intercept_init
        if isinstance(X, DesignMatrix):
            dm = X
        else:
            from superglm.solvers.pirls import _wrap_dense_X

            dm = _wrap_dense_X(X, groups)

        n = dm.n
        p = dm.p
        gms = dm.group_matrices

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError("weights must match the design row count")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("weights must contain at least one positive value")
        objective_merit_scale = _irls_objective_scale(
            y=self.y,
            weights=weights,
            family=family,
            link=link,
        )

        structured_decision = resolve_structured_backend(
            gms,
            groups,
            direct_solve=self.direct_solve,
            coefficient_width=p,
            lambda2=self.lambda2,
            S_override=S_override,
            nesting_cache=getattr(dm, "_structured_layout_cache", None),
        )
        if (
            structured_decision.use_structured
            and S_override is None
            and self.reml_penalties is None
        ):
            reason = (
                "the structured candidate requires compact reml_penalties "
                "when S_override is not supplied"
            )
            if self.direct_solve == "structured":
                raise ValueError(f"direct_solve='structured' is ineligible: {reason}.")
            structured_decision = replace(
                structured_decision,
                use_structured=False,
                fallback_reason=reason,
            )
        _use_structured = structured_decision.use_structured
        _direct_fallback_reason = structured_decision.fallback_reason
        # The structured step is chosen with its route, from the structure alone,
        # because its layout fixes the state's centre; QR or Gram is chosen in
        # ``_setup_systems``.
        _state_center: NDArray | None = None
        if _use_structured:
            self.step = step = _IRLSDirectStructuredStep(dm, groups, structured_decision)
            _state_center = step.state_centre(weights, p)

        if offset is None:
            offset = np.zeros(n)

        beta = self.beta_init.copy() if self.beta_init is not None else np.zeros(p)

        if intercept_init is not None:
            intercept = intercept_init
        else:
            intercept = coefficient_initial_intercept(
                distribution=family,
                link=link,
                y=self.y,
                sample_weight=weights,
            )
        # Whether the default start below was lowered into the family's mean space
        # (binomial/log; a start raised into an inverse link's domain is not).
        _start_lowered = False
        if intercept_init is None and (
            mean_space_violation(family, link) is not None
            or link_domain_violation(link, family) is not None
        ):
            # The domain guards below keep every accepted state inside the mean
            # space and the link's domain only from a start inside them, and the
            # default intercept is chosen before the offset: move it until the
            # start is inside (``irls_state.interior_start_intercept``).  A
            # caller's warm start is moved into the link's domain below.
            start_intercept = interior_start_intercept(
                family,
                link,
                (dm.matvec(beta) if np.any(beta) else 0.0) + intercept + offset,
                weights,
                intercept,
                level=intercept,
            )
            if start_intercept != intercept:
                intercept = start_intercept
                self._deviance_init = None
                _start_lowered = mean_space_violation(family, link) is not None
        elif link_domain_violation(link, family) is not None:
            # A warm start outside the link's domain is raised the same way, to
            # the default start's level ``g(mean)``. Step halving keeps states
            # inside only from a start inside (Marschner 2011), and a clipped row
            # outside carries a Fisher weight no solve can use. A warm start that
            # carries its centred state is read in it (design §3.8), not by
            # cancelling X beta against the raw intercept.
            _centred_init = self._centred_init
            if _centred_init is not None and np.shape(_centred_init[1]) == beta.shape:
                warm_alpha, warm_centre = _centred_init
                start_eta = (
                    float(warm_alpha)
                    + centred_matvec(dm, beta, np.asarray(warm_centre, dtype=np.float64))
                    + offset
                )
            else:
                start_eta = dm.matvec(beta) + intercept + offset
            if link_domain_violation(link, family)(start_eta, weights):
                intercept = interior_start_intercept(
                    family,
                    link,
                    start_eta,
                    weights,
                    intercept,
                    level=coefficient_initial_intercept(
                        distribution=family, link=link, y=self.y, sample_weight=weights
                    ),
                )
                self._deviance_init = None

        # Dense paths retain the existing p x p penalty oracle. Structured paths
        # add each penalty directly to A or d, unless a caller already supplied a
        # dense override (which remains authoritative).
        S: NDArray | None
        if _use_structured:
            S = None if S_override is None else np.asarray(S_override, dtype=np.float64)
        elif S_override is not None:
            S = S_override
        else:
            S = _build_penalty_matrix(
                gms, groups, self.lambda2, p, reml_penalties=self.reml_penalties
            )
        self.dm, self.n, self.p, self.gms, self.weights = dm, n, p, gms, weights
        self.offset, self.objective_merit_scale = offset, objective_merit_scale
        self.structured_decision, self._use_structured = structured_decision, _use_structured
        self._direct_fallback_reason, self._state_center = _direct_fallback_reason, _state_center
        self.beta, self.intercept, self._start_lowered, self.S = beta, intercept, _start_lowered, S
        # diag(S), formed once per call on first use by the mode score (S is fixed)
        self._penalty_curvature: list[NDArray] = []

    def penalty_matvec(self, beta_values: NDArray, *, magnitude: bool = False) -> NDArray:
        """Apply the fitted penalty without expanding an identity random-effect block.

        With ``magnitude``, return ``|S| |beta|``, the magnitude this product
        sums, which bounds its rounding componentwise.
        """
        S, lambda2 = self.S, self.lambda2
        values = np.asarray(beta_values, dtype=np.float64)
        if magnitude:
            values = np.abs(values)
        if S is not None:
            return (np.abs(S) if magnitude else S) @ values
        if self.reml_penalties is None:  # pragma: no cover - validated above
            raise RuntimeError("Structured penalty components are unavailable.")
        from superglm.reml.penalty_algebra import (
            penalty_component_magnitude_matvec,
            penalty_component_matvec,
        )

        apply = penalty_component_magnitude_matvec if magnitude else penalty_component_matvec
        product = np.zeros_like(values)
        for component in self.reml_penalties:
            lam = float(lambda2[component.name]) if isinstance(lambda2, dict) else float(lambda2)
            if lam == 0.0:
                continue
            product[component.group_sl] += lam * apply(
                component,
                values[component.group_sl],
                self.gms[component.group_index],
            )
        return product

    def penalty_rounding(self) -> int:
        """``k`` with ``fl(S v)`` within ``gamma_k fl(|S| |v|)`` (``penalty_product_rounding``)."""
        from superglm.reml.penalty_algebra import penalty_product_rounding

        return penalty_product_rounding(self.p, None if self.S is not None else self.reml_penalties)

    def penalty_curvature(self) -> NDArray:
        """``diag(S)``: the dense penalty's diagonal, or the components' without forming ``S``."""
        lambda2, reml_penalties = self.lambda2, self.reml_penalties
        if self.S is not None:
            return np.diag(self.S).astype(np.float64, copy=True)
        from superglm.reml.identified import penalty_diagonal

        lambdas = (
            lambda2
            if isinstance(lambda2, dict)
            else {component.name: float(lambda2) for component in reml_penalties or ()}
        )
        return penalty_diagonal(self.p, lambdas, reml_penalties)

    def penalty_quadratic(self, beta_values: NDArray) -> float:
        values = np.asarray(beta_values, dtype=np.float64)
        return float(values @ self.penalty_matvec(values))

    def mode_residual(
        self,
        beta_values: NDArray,
        intercept_value: float,
        mu_values: NDArray,
        eta_values: NDArray,
        centred_intercept: float | None = None,
        resolve_cap: float = MODE_RESOLVE_CAP,
    ) -> ModeResidual:
        """The observed-REML certificate's score at an iterate (``convergence="mode_score"``).

        The row score from Fisher rows (the score itself does not depend on
        the curvature the rows carry), centred on the system the last solve
        linearised about (``_score_centre``), its floors and weak tests
        resolved once every relative score is within ``MODE_RESOLVE_CAP``
        (``solvers.mode_score``).  The floors read the iterate's intercept
        about ``mean_x``; a centred state reads it from its own ``alpha`` and
        the offset of ``mean_x`` from the centre (``_system_offset_mean``), not
        from the raw intercept, which cancels ``c' beta`` at a column's offset.
        """
        _penalty_curvature = self._penalty_curvature
        assert self._score_centre is not None
        mean_x, sum_w, diagonal, excluded_indices, offset_mean = self._score_centre
        rows = coefficient_working_rows(
            distribution=self.family,
            link=self.link,
            y=self.y,
            mu=mu_values,
            eta=eta_values,
            sample_weight=self.weights,
            prefer_observed=False,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt(np.abs(diagonal) / sum_w)
        excluded = np.zeros(self.p, dtype=bool)
        excluded[list(excluded_indices)] = True
        # the slopes the REML fit's Laplace approximation leaves out
        # (``reml.identified``, design §3.9): flagged and kept, never gated
        excluded[list(self._laplace_excluded)] = True
        if centred_intercept is None or offset_mean is None:
            shift = float(mean_x @ beta_values)
            alpha = float(intercept_value) + shift
            eta_tilde = eta_values - self.offset - float(intercept_value) - shift
        else:
            shift = float(offset_mean @ beta_values)
            alpha = float(centred_intercept) + shift
            eta_tilde = eta_values - self.offset - float(centred_intercept) - shift
        if not _penalty_curvature:
            _penalty_curvature.append(self.penalty_curvature())
        return penalized_mode_residual(
            dm=self.dm,
            row_score=rows.weights * (rows.response - eta_values),
            fisher_weights=rows.weights,
            positive_prior=self.weights > 0.0,
            mean_x=mean_x,
            centered_scale=np.where(np.isfinite(scale), scale, 0.0),
            alpha=alpha,
            eta_tilde=eta_tilde,
            penalty_score=self.penalty_matvec(beta_values),
            penalty_magnitude=self.penalty_matvec(beta_values, magnitude=True),
            penalty_curvature=_penalty_curvature[0],
            sum_w=sum_w,
            bar=self.mode_bar,
            excluded=excluded,
            resolve_cap=resolve_cap,
            penalty_rounding=self.penalty_rounding(),
        )

    def evaluate_state(
        self,
        beta_values: NDArray,
        intercept_value: float,
        *,
        phase: str,
        iteration: int,
        alpha: float | None = None,
        deviance: float | None = None,
        eta_unclipped: NDArray | None = None,
        enclosing_proposal_state_id: int | None = None,
        emit_trace: bool = True,
        centred_intercept: float | None = None,
    ) -> _IRLSState:
        recorder, _state_center = self.recorder, self._state_center
        if recorder.trace_enabled:
            trace_run = recorder.trace_run
            assert trace_run is not None
            state_id = trace_run.next_state_id()
            evaluation_id = trace_run.next_evaluation_id()
        else:
            state_id = None
            evaluation_id = None
        if _state_center is not None:
            # One-engine design §3.8: the state is (alpha, beta) about the fixed
            # prior-weighted centre, eta = alpha + X~ beta + offset, and the raw
            # intercept is only its reading; a state entering from raw
            # coordinates (the start, a warm start) is centred once.
            shift = math.fsum(_state_center * np.asarray(beta_values, dtype=np.float64))
            if centred_intercept is None:
                centred_intercept = float(intercept_value) + shift
            if eta_unclipped is None:
                started = time.perf_counter()
                eta_unclipped = (
                    centred_intercept
                    + centred_matvec(self.dm, beta_values, _state_center)
                    + self.offset
                )
                recorder.eta_formed(time.perf_counter() - started)
            intercept_value = centred_intercept - shift
        state = _evaluate_irls_state(
            self.dm,
            self.y,
            self.weights,
            self.family,
            self.link,
            self.offset,
            beta_values,
            intercept_value,
            deviance=deviance,
            eta_unclipped=eta_unclipped,
            state_id=state_id,
            evaluation_id=evaluation_id,
            basis_id=recorder.trace_basis_id,
            lambdas=recorder.resolved_lambdas,
            eta_clock=recorder.eta_formed,
        )
        if not self._has_scop:
            state = replace(
                state,
                penalized_deviance=float(state.deviance + self.penalty_quadratic(state.beta)),
            )
        if centred_intercept is not None:
            state = replace(state, centred_intercept=float(centred_intercept))
        if emit_trace:
            recorder.emit_evaluation(
                state,
                phase=phase,
                iteration=iteration,
                alpha=alpha,
                enclosing_proposal_state_id=enclosing_proposal_state_id,
                deviance_reused=deviance is not None,
            )
        return state

    def _setup_routes(self) -> None:
        family, link, groups = self.family, self.link, self.groups
        _state_center, compute_rank_info = self._state_center, self.compute_rank_info
        _return_working_system = self._return_working_system
        # ── Constrained QP support (monotone splines) ──
        # The step with linear constraints is chosen with its route: it builds
        # the constraint system, fixed for the whole fit, which the feasibility
        # contract reads whatever the step (``qp``; ``None`` without constraints).
        has_constraints = any(g.constraints is not None for g in groups)
        qp = _IRLSDirectQPStep(groups, self.p, family, link) if has_constraints else None
        self.qp = qp
        if qp is not None:
            self.step = qp

        # ── SCOP monotone engine support ──
        _has_scop = any(g.monotone_engine == "scop" for g in groups)
        self._scop_run: SCOPRunCentring | None = None
        if (
            _state_center is None
            and not self._use_structured
            and not _has_scop
            and hasattr(self.dm, "group_matrices")
        ):
            # The nested chain's centred state on gram's centred system too: eta
            # about the fixed prior-weighted centre, so a column at 1e8
            # contributes rows of its spread, not of its offset, and the
            # objective the line search compares keeps its digits (stage-1
            # verifier: a raw-1e8 column left the penalized deviance noisy at
            # 1e-11 relative, so no Newton step near the mode was accepted and
            # the score stalled at 1e-6).
            _state_center = prior_weighted_centre(self.dm, self.weights)
        _scop_curvature = "fisher"
        if _has_scop:
            from superglm.reml.observed_geometry import classify_scop_reml_curvature

            _scop_curvature = classify_scop_reml_curvature(family, link)
        if compute_rank_info and self._defer_rank_info is not None:
            raise ValueError("deferred rank metadata requires compute_rank_info=False")
        if (not self._compute_fit_statistics and compute_rank_info) or (
            _return_working_system and (compute_rank_info or has_constraints or _has_scop)
        ):
            raise ValueError("intermediate REML shortcuts require rank metadata to be disabled")
        if not self._compute_reml_geometry and (compute_rank_info or self._compute_fit_statistics):
            raise ValueError(
                "omitting generic REML geometry requires rank metadata and fit statistics "
                "to be disabled"
            )
        # The curvature of the working rows, decided here once from the family,
        # the link and the route, and switched at most once by the stop rule.
        self.curvature = _IRLSDirectCurvaturePolicy(
            family,
            link,
            self.y,
            self.weights,
            use_observed_newton=self._use_observed_newton,
            working_newton=self._working_newton,
            return_working_system=_return_working_system,
            latch=self._newton_latch,
            has_constraints=has_constraints,
            has_scop=_has_scop,
            use_structured=self._use_structured,
        )
        self._has_scop = _has_scop
        self._state_center = _state_center
        if _has_scop:
            # The step is chosen with its route.  Its setup evaluates the start,
            # so it follows every field the evaluation reads.
            self.step = _IRLSDirectSCOPStep(self, _scop_curvature)

    def _setup_systems(self) -> None:
        dm, family, link, groups, gms = self.dm, self.family, self.link, self.groups, self.gms
        has_constraints, _has_scop = self.qp is not None, self._has_scop
        _use_structured, _raw_moment_policy = self._use_structured, self._raw_moment_policy
        _fisher_data_reuse = self._fisher_data_reuse
        # The fit's one step method, chosen here unless it is structured
        # (``_setup_design``), QP or SCOP (``_setup_routes``): QR materialises the
        # full design once.
        # Constrained QP / SCOP requires Gram path — force it if constraints present;
        # a design with no columns has nothing to factor, and the Gram path solves
        # its intercept
        _use_qr = self.direct_solve == "qr" and self.p > 0 and not has_constraints and not _has_scop
        # The binomial/log mean space's overlay, ``None`` for a family and link
        # whose means cannot leave their space: its certificate judges every
        # route, and its Newton steps are available off the QR route (whose
        # least squares cannot carry a zero-curvature row's score), the SCOP
        # route (which takes its own Newton steps) and the fREML performance
        # iteration.
        self.mean_space = (
            None
            if mean_space_violation(family, link) is None
            else _IRLSDirectMeanSpaceOverlay(
                available=not (_use_qr or _has_scop or self._return_working_system),
                start_lowered=self._start_lowered,
            )
        )
        if not (_use_structured or has_constraints or _has_scop):
            self.step: _IRLSDirectStep = (
                _IRLSDirectQRStep(gms, self.S, self.p) if _use_qr else _IRLSDirectDenseStep()
            )

        # Tabmat acceleration: the structured layout owns a pruned small-block
        # plan, so it must not construct a split containing the dominant factor.
        # Other non-discrete paths retain the shared full-design split behavior.
        if _use_structured:
            _tabmat_split = None
        elif _use_qr:
            _tabmat_split = None
        elif has_constraints:
            # Constrained QP uses the shared stable centered-system builder, while
            # retaining the execution-plan route instead of materializing a
            # separate tabmat copy of the design.
            _tabmat_split = None
        else:
            # Ordinary intercept profiling currently benefits only when the split
            # contains a native high-cardinality categorical component. Avoid
            # materializing an unused dense duplicate for numeric and low-cardinality fits.
            _tabmat_split = dm.tabmat_centering_split
        _can_reuse_weighted_gram = self.curvature.weighted_gram_reuse
        dm.execution_plan.validate_group_spans(groups)
        _defer_raw_spline = (
            not dm.raw_spline_tabmat_plan_built
            and _is_raw_spline_tabmat_centering_candidate(gms, n=self.n)
            and (
                _defer_raw_spline_tabmat_plan(
                    n=self.n,
                    raw_width=sum(int(group.B.shape[1]) for group in gms if group.shape[1] > 0),
                    constant_weights=_can_reuse_weighted_gram,
                    repeated_fit=self.trace_purpose
                    in {"reml_bootstrap", "reml_candidate", "reml_line_search"},
                )
            )
        )
        _tabmat_centering_state = TabmatCenteringState(
            raw_spline_eligible=False if _defer_raw_spline else None
        )
        if self._scop_run is not None:
            _seed_refusals(_tabmat_centering_state, self._scop_run.full_refusals)
        fixed_owners = ()
        if _raw_moment_policy is not None or _fisher_data_reuse is not None:
            # A negative route choice is safe for changed W; no accepted preflight,
            # weighted system or penalty survives. Fixed-coordinate QP is eligible.
            fixed_owners = (
                (dm, dm.execution_plan, family, link, *gms, *groups)
                if type(dm) is DesignMatrix
                and type(family) in (Poisson, Gaussian, Gamma, Binomial, NegativeBinomial, Tweedie)
                and type(link) in (LogLink, IdentityLink, LogitLink)
                and all(
                    type(gm)
                    in (
                        CategoricalGroupMatrix,
                        DenseGroupMatrix,
                        SparseSSPGroupMatrix,
                        SupportCompressedSSPGroupMatrix,
                    )
                    for gm in gms
                )
                and all(
                    type(group) is GroupSlice
                    and (
                        group.constraints is None or type(group.constraints) is LinearConstraintSet
                    )
                    for group in groups
                )
                and not (_use_qr or _use_structured or _has_scop)
                and self.debug_recorder is None
                and self.trace_run is None
                else ()
            )
            if _raw_moment_policy is not None:
                _tabmat_centering_state.raw_moment_eligible = _raw_moment_policy.seed_raw_rejection(
                    fixed_owners
                )
                if not fixed_owners:
                    _raw_moment_policy = None
        if _fisher_data_reuse is not None and not (
            fixed_owners and type(family) is Gamma and type(link) is LogLink and not has_constraints
        ):
            _fisher_data_reuse.clear()
            _fisher_data_reuse = None
        if _defer_raw_spline:
            self.recorder.count("centered_spline_tabmat_cold_policy_rejections")
        self._constant_centered_cache: CenteredSystem | None = None
        self._constant_centered_z: NDArray | None = None
        # the factor certificate and its one-entry cache (``_certify_centered_factor``)
        self.certify_centered_factor = partial(
            _certify_centered_factor, dm, _fisher_data_reuse, [None]
        )
        # the last Gram decomposition and the matrix it was taken of (``_authoritative_gram_rank``)
        self.authoritative_gram_rank = partial(_authoritative_gram_rank, [None])
        _initial_data_pending = (
            self._initial_data_reuse is not None
            and type(family) is Poisson
            and type(link) is LogLink
            and type(dm) is DesignMatrix
            and all(
                type(gm)
                in (
                    CategoricalGroupMatrix,
                    DenseGroupMatrix,
                    SparseSSPGroupMatrix,
                    SupportCompressedSSPGroupMatrix,
                )
                for gm in gms
            )
            and not (_use_qr or _use_structured or has_constraints or _has_scop)
            and self.debug_recorder is None
            and self.trace_run is None
        )
        # These objects stay live throughout the owner line search. A new design
        # or warm state cannot consume an entry from the previous coordinates.
        self._initial_data_key = (
            (
                id(dm),
                id(dm.execution_plan),
                id(family),
                id(link),
                self.beta.tobytes(),
                np.float64(self.intercept).tobytes(),
            )
            if _initial_data_pending
            else None
        )
        self._tabmat_split = _tabmat_split
        self._tabmat_centering_state = _tabmat_centering_state
        self.fixed_owners = fixed_owners
        self._raw_moment_policy, self._fisher_data_reuse = _raw_moment_policy, _fisher_data_reuse
        self._initial_data_pending = _initial_data_pending

    def get_centered_system(self, W_current: NDArray, z_off_current: NDArray) -> CenteredSystem:
        _tabmat_centering_state, curvature = self._tabmat_centering_state, self.curvature
        # read on every call: the Gamma/log switch turns it off mid-fit
        _can_reuse_weighted_gram = curvature.weighted_gram_reuse
        _constant_centered_cache = self._constant_centered_cache
        reuse = self._initial_data_reuse if self._initial_data_pending else None
        self._initial_data_pending = False
        before = replace(_tabmat_centering_state) if reuse is not None else None
        if reuse is not None:
            reused = reuse.take(
                self._initial_data_key,
                W_current,
                z_off_current,
                _tabmat_centering_state,
                np.asarray(self.S),
            )
            if reused is not None:
                return reused
        if (
            _can_reuse_weighted_gram
            and _constant_centered_cache is not None
            and self._constant_centered_z is not None
        ):
            if np.array_equal(z_off_current, self._constant_centered_z):
                return _constant_centered_cache
            self._constant_centered_cache = refresh_centered_rhs(
                system=_constant_centered_cache,
                dm=self.dm,
                W=W_current,
                z_off=z_off_current,
            )
            self._constant_centered_z = z_off_current.copy()
            return self._constant_centered_cache
        # reuse is off whenever the rows are observed (``_IRLSDirectCurvaturePolicy``)
        fisher = self._fisher_data_reuse if _can_reuse_weighted_gram else None
        data = (
            fisher.take((*self.fixed_owners, self.weight_semantics), W_current)
            if fisher is not None
            else None
        )
        system = build_centered_system(
            dm=self.dm,
            W=W_current,
            z_off=z_off_current,
            penalty=np.asarray(self.S),
            tabmat_split=self._tabmat_split,
            tabmat_state=_tabmat_centering_state,
            profile=self.profile,
            _data=data,
        )
        if fisher is not None and fisher.data is None:
            fisher.remember(W_current, system)
        if self._scop_run is not None:
            _carry_refusals(self._scop_run.full_refusals, _tabmat_centering_state)
        if (
            self._raw_moment_policy is not None
            and _tabmat_centering_state.raw_moment_eligible is False
        ):
            self._raw_moment_policy.raw_moment_eligible = False
        if reuse is not None:
            reuse.remember(
                self._initial_data_key,
                W_current,
                z_off_current,
                before,
                _tabmat_centering_state,
                system,
            )
        if _can_reuse_weighted_gram:
            self._constant_centered_cache = system
            self._constant_centered_z = z_off_current.copy()
        return system

    def _setup_loop(self) -> None:
        _state_center = self._state_center
        self.converged = False
        self._last_working_centered: CenteredSystem | None = None
        # its mean_x less the state's centre (``_system_offset_mean``), None without one
        self._last_working_offset_mean: NDArray | None = None
        # convergence="mode_score": the centring of the certificate's score at the
        # iterate each solve linearised about -- (mean_x, sum_w, centred diagonal,
        # coefficients the factor truncated as weakly identified, mean_x less the
        # state's centre or None without one) -- or None on a
        # route that forms no centred system (SCOP, linear constraints), which
        # keeps the coefficient-step test.
        self._score_centre: tuple | None = None
        # the certificate's bar (``mode_score.mode_certification_bar``): the REML
        # fit passes the one its stopping tolerance needs
        mode_bar = MODE_CERTIFICATION_BAR if self._mode_bar is None else float(self._mode_bar)
        self.stop_rule = _IRLSDirectConvergence(self.max_iter, mode_bar)
        # the shared line search, unless the step searches its own way
        self.trial_search = _IRLSDirectTrialSearch() if self.step.search is None else None
        # A warm start's centred state carried to this fit's centre (one-engine
        # design §3.8): alpha + (X - 1 c_warm') beta = alpha_c + (X - 1 c') beta
        # gives alpha_c = alpha + fsum((c - c_warm) beta), exact on every column
        # the two centres share.  Taken back from the raw intercept instead it
        # errs by u |c' beta| in every row's eta, ~0.25 at a 1e16 offset.
        centred_start: float | None = None
        if (
            self._centred_init is not None
            and _state_center is not None
            and self.beta_init is not None
            and self.intercept_init is not None
            and self.intercept == self.intercept_init
        ):
            warm_alpha, warm_centre = self._centred_init
            warm_centre = np.asarray(warm_centre, dtype=np.float64)
            if warm_centre.shape == _state_center.shape:
                centred_start = float(warm_alpha) + math.fsum(
                    (_state_center - warm_centre) * self.beta
                )
        # Freeze the fit-entry state so iteration-one trial safety has a baseline.
        # A step with its own entry state traces that state, not this one.
        initial_state = self.step.initial_state
        committed = self.evaluate_state(
            self.beta,
            self.intercept,
            phase="initial",
            iteration=0,
            deviance=self._deviance_init,
            emit_trace=initial_state is None,
            centred_intercept=centred_start,
        )
        if initial_state is not None:
            committed = initial_state(self, committed)
        self.recorder.emit_state_commit(
            committed,
            iteration=0,
            fit_converged=False,
            convergence_value=None,
            termination_reason=None,
        )
        self.eta_unclipped = committed.eta_unclipped
        self.eta = committed.eta
        self.mu = committed.mu

        self.max_halving = 20  # max step-halving attempts per iteration
        self._consecutive_svd = 0  # for auto-mode warning
        # Declared, not bound: the stop rule's chain (``_IRLSDirectConvergence``)
        # and the final refusals are the whole set of reasons this solver can end
        # on, and declaring them against the shared vocabulary is what ties the
        # two together. A tenth reason invented there is a type error on its own
        # assignment until ``TerminationReason`` carries it, so the consumers that
        # switch on the field cannot be handed a string they have never been shown.
        self.termination_reason: TerminationReason
        self.mode_bar, self.committed = mode_bar, committed

    def _working(self, it: int) -> None:
        committed, family, link = self.committed, self.family, self.link
        recorder, mean_space = self.recorder, self.mean_space
        self.rank_truncated: bool | None = None
        self.used_rank_certification = False
        self.proposal_centred_intercept: float | None = None

        # Working quantities from current eta/mu (already computed)
        _t0 = time.perf_counter()
        working_rows = self.curvature.working_rows(self.mu, self.eta)
        W = working_rows.weights
        z = working_rows.response
        # the mean space's Newton rows once engaged, in score form
        # (``_IRLSDirectMeanSpaceOverlay.newton_rows``)
        newton_score: NDArray | None = None
        if mean_space is not None and mean_space.engaged:
            newton_rows = mean_space.newton_rows(self, committed)
            if newton_rows is not None:
                W, newton_score, z = newton_rows
        if working_rows.rejection_reason is not None:
            # Every accepted trial of an observed fit had finite rows (the
            # line search rejects the others), so only the entry state can
            # get here: the declared curvature has no step to take.
            raise ValueError(
                "IRLS direct cannot start: the observed-Newton rows of "
                f"{type(family).__name__} with a {type(link).__name__} are not finite at "
                "the starting coefficients."
            )
        if working_rows.curvature_source == "observed":
            recorder.count("irls_observed_newton_iters")
        self.working_eta_unclipped = self.eta_unclipped
        self.working_eta = self.eta
        self.working_mu = self.mu
        # the separation backstop and the diagnostics read Fisher's weights,
        # the curvature their bars were set on, on a Newton iteration too
        positive_w_min, positive_w_max, w_ratio = _positive_working_weight_stats(
            working_rows.weights
        )
        recorder.t_working += time.perf_counter() - _t0

        if w_ratio > 1e12:
            logger.debug(
                "IRLS direct iter=%d: extreme W ratio %.1e (positive W range [%.2e, %.2e])",
                it + 1,
                w_ratio,
                positive_w_min,
                positive_w_max,
            )
        self.working_rows, self.W, self.z, self.newton_score = working_rows, W, z, newton_score
        self.w_ratio = w_ratio

    def _finish(self, it: int) -> None:
        retained, family, link, beta = self.retained, self.family, self.link, self.beta
        weights, eta_unclipped, eta, mu = self.weights, self.eta_unclipped, self.eta, self.mu
        dev, converged, qp = self.dev, self.converged, self.qp
        termination_reason = self.termination_reason

        # A published Gaussian identity fit carries its centred intercept as the
        # compensated pair (alpha, alpha_lo): alpha's weighted mean rounds by a
        # kernel-dependent ulp or more, and alpha + (x - c) beta ties at every row
        # when alpha* is the midpoint of two adjacent floats, so a one-ulp signal
        # between two levels collapsed on some BLAS kernels.  The deviance and
        # scale published below are the compensated predictor's, the eta every
        # consumer reads (``mode_score.linear_predictor``).
        centred_intercept_lo: float | None = None
        if (
            self._compensate_centred_intercept
            and self._compute_fit_statistics
            and retained.centred_intercept is not None
            and self._state_center is not None
            and type(family) is Gaussian
            and type(link) is IdentityLink
        ):
            contribution = centred_matvec(self.dm, beta, self._state_center)
            centred_intercept_lo = centred_intercept_remainder(
                self.y, weights, self.offset, retained.centred_intercept, contribution
            )
            if centred_intercept_lo is not None:
                contribution += centred_intercept_lo
                eta_unclipped = (retained.centred_intercept + contribution) + self.offset
                eta = stabilize_eta(eta_unclipped, link)
                mu = clip_mu(link.inverse(eta), family)
                dev = weighted_row_sum(weights, family.deviance_unit, self.y, mu)

        # Runtime separation backstop (issue #341).  Two terminal signatures mark
        # a coefficient that walked toward +/-infinity instead of converging:
        #
        # * the retained linear predictor is pinned at the link's overflow guard
        #   on rows that carry weight -- the guard, not the likelihood, stopped
        #   the walk, and it also manufactures "convergence" by freezing the
        #   coefficient once eta saturates;
        # * the budget ran out with a frozen deviance while the coefficient
        #   criterion still failed -- the field trace of issue #340/#341.
        #
        # Both require the extreme working-weight range that a drifting cell
        # produces.  The build-time scan names cell separation before fitting
        # starts; this catches what the design scan cannot see (non-categorical
        # indicator structure), promoting a debug-level log line to a named
        # warning or, in strict mode, an error.
        if self.separation != "ignore" and np.isfinite(dev):
            from superglm.diagnostics.separation import (
                EXTREME_WEIGHT_RATIO,
                STAGNANT_DEVIANCE_DELTA,
                SeparationError,
                SeparationWarning,
                drifting_groups,
                format_runtime_message,
            )

            separation_ratio = _separation_weight_ratio(
                self.working_rows.curvature_source,
                self.w_ratio,
                family=family,
                link=link,
                mu=self.working_mu,
                eta=self.working_eta,
                weights=weights,
            )
            if separation_ratio > EXTREME_WEIGHT_RATIO:
                pinned = bool(np.any((eta != eta_unclipped) & (weights > 0)))
                exhausted_stagnant = (
                    not converged
                    # A warm-started micro-budget solve (REML performance
                    # iterations run max_iter=1) exhausts its budget by
                    # construction; only a real budget makes exhaustion-with-
                    # stagnation evidence of a drift (the field trace ran 100).
                    and self.max_iter >= 10
                    and it + 1 >= self.max_iter
                    and not self.step_rejected
                    and self.deviance_relative_change < STAGNANT_DEVIANCE_DELTA
                )
                if pinned or exhausted_stagnant:
                    drifting = drifting_groups(
                        self.dm, self.groups, beta, eta, eta_unclipped, weights
                    )
                    message = format_runtime_message(separation_ratio, it + 1, drifting, pinned)
                    if self.separation == "error":
                        raise SeparationError(message)
                    # fit_irls_direct's frame: _finish, run, _fit_irls_direct_once, it
                    held_warnings.warn(message, SeparationWarning, stacklevel=4)

        if qp is not None:
            if not qp.feasible(beta):
                minimum_scaled_slack = float(np.min(_feasibility_slack(qp.A_all, beta, qp.b_all)))
                converged = False
                termination_reason = "constraint_infeasible"
                logger.warning(
                    "fit_irls_direct: retained coefficient mode violates hard constraints "
                    "(minimum scaled slack %.3e, tolerance %.1e); fit is not converged.",
                    minimum_scaled_slack,
                    _QP_FEASIBILITY_TOL,
                )
            elif not qp.retained_converged:
                converged = False
                termination_reason = "constraint_kkt_incomplete"
                logger.warning(
                    "fit_irls_direct: retained coefficient mode has no complete "
                    "constrained-QP KKT certificate; fit is not converged."
                )

        # A returned state with a row outside the link's domain is no fitted model,
        # whatever stopped the loop: the state a shape-constrained start leaves it
        # at when no step can move it in (``link_domain_violation``).
        outside_link_domain = link_domain_violation(link, family)
        if outside_link_domain is not None and outside_link_domain(eta_unclipped, weights):
            from superglm.diagnostics.convergence import ConvergenceWarning

            if converged:
                converged, termination_reason = False, "score_stagnated"
            # fit_irls_direct's frame: _finish, run, _fit_irls_direct_once, it
            held_warnings.warn(
                "The coefficient fit returns its last iterate, not converged: some rows' "
                f"linear predictor lies outside the range the {type(link).__name__} allows "
                f"a {type(family).__name__} mean. This link may not suit these data; a log "
                "link cannot leave that range.",
                ConvergenceWarning,
                stacklevel=4,
            )

        # the mean space's boundary verdict on the returned state (F04)
        mean_space = self.mean_space
        if mean_space is not None:
            converged, termination_reason = mean_space.final(
                self, eta_unclipped, converged, termination_reason
            )
        self.centred_intercept_lo = centred_intercept_lo
        self.eta_unclipped, self.eta, self.mu, self.dev = eta_unclipped, eta, mu, dev
        self.converged, self.termination_reason = converged, termination_reason

    def _terminal_rows(self) -> None:
        W, z, terminal = self.W, self.z, self.step.terminal
        # Every public/exported matrix and rank claim is evaluated at the retained
        # model. Private fREML performance iterations deliberately reuse the
        # working system used for their one coefficient update.
        if not self._return_working_system:
            export_rows = coefficient_working_rows(
                distribution=self.family,
                link=self.link,
                y=self.y,
                mu=self.mu,
                eta=self.eta,
                sample_weight=self.weights,
                prefer_observed=False,
            )
            W = export_rows.weights
            z = export_rows.response
            self.export_rows = export_rows
        self.W, self.z = W, z
        if terminal is not None:
            # the step's own state at the retained model (the SCOP Hessian caches,
            # from the export rows)
            terminal(self)

    def _publish(
        self, it: int, final: _IRLSDirectFinalGeometry
    ) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
        _compute_fit_statistics, family, y = self._compute_fit_statistics, self.family, self.y
        weights, p_eff, retained = self.weights, final.p_eff, self.retained
        XtWX_S_inv_beta, mean_space = final.XtWX_S_inv_beta, self.mean_space
        _resolved_direct_backend = self.step.backend
        self.recorder.disclose_backend(self, _resolved_direct_backend, final.structured_factor)

        # Pearson-based phi for estimated-scale families.  The numerator is the
        # same under either weight contract; only the denominator's likelihood
        # size distinguishes them.
        if _compute_fit_statistics and not getattr(family, "scale_known", True):
            pearson_sum = pearson_chi2(distribution=family, y=y, mu=self.mu, sample_weight=weights)
            df_resid = pearson_residual_degrees_of_freedom(
                weights,
                p_eff,
                weight_semantics=self.weight_semantics,
            )
            phi = pearson_sum / df_resid
        else:
            phi = 1.0
        if not self._compute_reml_geometry or not _compute_fit_statistics:
            # Private SCOP candidates and stats-skipping in-loop REML fits have no
            # retained-fit statistics. NaN makes accidental publication fail
            # visibly instead of presenting 0 EDF or unit dispersion as if either
            # had been evaluated; published statistics come from the terminal
            # refit, which always computes them.
            p_eff = float("nan")
            phi = float("nan")

        result = PIRLSResult(
            beta=self.beta,
            intercept=self.intercept,
            n_iter=it + 1,
            deviance=self.dev,
            converged=self.converged,
            phi=phi,
            effective_df=p_eff,
            iteration_log=self.recorder.iteration_log if self.record_diagnostics else None,
            log_det_H=final.log_det_H,
            reml_hessian_rank=final.reml_hessian_rank,
            reml_geometry=final.reml_geometry_summary,
            reml_slope_decomposition=final.retained_reml_decomposition,
            rank_info=final.rank_info,
            state_id=retained.state_id,
            evaluation_id=retained.evaluation_id,
            state_space=retained.state_space,
            basis_id=retained.basis_id,
            termination_reason=self.termination_reason,
            direct_backend=_resolved_direct_backend,
            direct_fallback_reason=self._direct_fallback_reason,
            centred_intercept=retained.centred_intercept,
            state_center=None if retained.centred_intercept is None else self._state_center,
            centred_intercept_lo=self.centred_intercept_lo,
            mean_space_boundary_rows=0 if mean_space is None else mean_space.boundary_rows,
            mean_space_true_mode=bool(
                self.converged and mean_space is not None and mean_space.true_confirmed
            ),
            truncated_directions=() if mean_space is None else mean_space.published_truncated,
            mean_space_boundary_unresolved=(
                mean_space is not None and mean_space.truncated_boundary
            ),
        )

        # the step's part of the publication: the SCOP latent states and inference
        postfit = self.step.postfit
        scop_converged = None if postfit is None else postfit(self, result, final)

        if self.return_xtwx:
            if self.return_scop_state and scop_converged is not None:
                return result, XtWX_S_inv_beta, final.XtWX_beta, scop_converged
            return result, XtWX_S_inv_beta, final.XtWX_beta

        if self.return_scop_state and scop_converged is not None:
            return result, XtWX_S_inv_beta, scop_converged
        return result, XtWX_S_inv_beta
