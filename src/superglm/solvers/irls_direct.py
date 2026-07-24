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
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

import superglm.solvers.scop_exact_support as scop_exact_support
from superglm._fit_trace import TraceRun
from superglm._group_matrix._group_matrix_centered import (
    _raw_centering_well_scaled,
    stable_centered_matvec,
)
from superglm._group_matrix._group_matrix_tabmat import (
    _defer_raw_spline_tabmat_plan,
    _is_raw_spline_tabmat_centering_candidate,
)
from superglm.distributions import _VARIANCE_FLOOR, Distribution, initial_mean
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
)
from superglm.links import Link
from superglm.solvers.centered_system import (
    CenteredSystem,
    TabmatCenteringState,
    build_anchor_centered_system,
    build_centered_system,
    grouped_augmented_factor,
    grouped_augmented_factor_rhs,
    grouped_weighted_factor,
    refresh_centered_rhs,
)
from superglm.solvers.constrained_qp import solve_constrained_qp
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom
from superglm.solvers.hessian_factor import HessianFactor
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
    RankDecomposition,
    RankInfo,
    decompose_factor,
    decompose_gram,
    decompose_symmetric,
    needs_factor_certification,
)
from superglm.solvers.scop import SCOPSolverReparam
from superglm.solvers.scop_newton import scop_joint_newton_step, scop_newton_step
from superglm.solvers.structured import (
    ProfiledScalarSchurFactor,
    ScalarStructuredSystem,
    SymmetricBlockOperator,
    build_augmented_scalar_factor,
    build_penalized_scalar_operator,
    build_scalar_structured_system,
    get_scalar_structured_layout,
    resolve_structured_backend,
)
from superglm.solvers.working_rows import (
    coefficient_working_rows,
    supports_observed_newton,
)
from superglm.types import GroupSlice, PenaltyComponent

logger = logging.getLogger(__name__)


def _stable_penalized_deviance_delta(
    candidate: _IRLSState,
    committed: _IRLSState,
    penalty_matvec: Callable[[NDArray], NDArray],
) -> float:
    """Compare ``D + beta' S beta`` without subtracting two quadratics.

    In an ill-conditioned smooth basis, the two penalty quadratics can each be
    accurately evaluated while their tiny difference loses enough digits to
    reverse the sign of an otherwise safe terminal step.  The polarization
    identity evaluates that difference directly from the coefficient update.
    """
    delta_beta = candidate.beta - committed.beta
    penalty_direction = penalty_matvec(candidate.beta + committed.beta)
    penalty_delta = math.fsum(
        float(delta_value * direction_value)
        for delta_value, direction_value in zip(
            delta_beta,
            penalty_direction,
            strict=True,
        )
    )
    return float(
        math.fsum(
            (
                float(candidate.deviance),
                -float(committed.deviance),
                penalty_delta,
            )
        )
    )


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


@dataclass(frozen=True)
class _CenteredFactorCertification:
    """One fit-local factor certificate tied to immutable centered geometry."""

    system: CenteredSystem
    factor: NDArray
    decomposition: RankDecomposition
    transformed_rhs: NDArray | None


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
    state_id: int | None = None,
    evaluation_id: int | None = None,
    basis_id: int | None = None,
    lambdas: tuple[tuple[str, object], ...] = (),
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

    irls = _evaluate_irls_state(
        dm,
        y,
        weights,
        family,
        link,
        offset,
        beta_trial,
        intercept_trial,
        state_id=state_id,
        evaluation_id=evaluation_id,
        basis_id=basis_id,
        lambdas=lambdas,
    )
    return _SCOPTrialState(irls=irls, groups=tuple(trial_groups))


def _has_constant_irls_weights(family: Distribution, link: Link) -> bool:
    """Return True when PIRLS weights are independent of ``mu``.

    The direct solver can reuse X'WX only when
    ``(dmu/deta)^2 / V(mu)`` is exactly constant.  Keep this deliberately
    conservative so performance never changes the fitted problem.
    """
    from superglm.distributions import Gamma, Gaussian
    from superglm.links import IdentityLink, LogLink

    return (type(family) is Gaussian and type(link) is IdentityLink) or (
        type(family) is Gamma and type(link) is LogLink
    )


def _working_sums(W: NDArray, Wz: NDArray) -> tuple[float, float]:
    """Validate the already-required intercept sums before moment kernels."""
    with np.errstate(over="ignore", invalid="ignore"):
        sum_w = float(np.sum(W, dtype=np.float64))
        sum_wz = float(np.sum(Wz, dtype=np.float64))
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise ValueError("working weights must have a positive finite sum")
    if not np.isfinite(sum_wz):
        raise ValueError("weighted working response must have a finite sum")
    return sum_w, sum_wz


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
    _compute_reml_geometry: bool = True,
    _use_observed_newton: bool = True,
    _deviance_init: float | None = None,
    trace_run: TraceRun | None = None,
    trace_purpose: str = "fit",
    _compute_scop_postfit_inference: bool = True,
) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
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
    _compute_reml_geometry : bool
        Internal SCOP-candidate switch. If False, omit the generic profiled
        slope inverse, determinant, and rank because the caller replaces them
        with one joint latent-coordinate LAML geometry. This requires both
        retained rank metadata and fit statistics to be disabled. Public and
        terminal fits must leave this True.
    _use_observed_newton : bool
        Internal rescue-controller switch. When enabled, an ordinary Gamma/log
        fit switches to its exact positive observed curvature only after an
        atomic Fisher proposal rejection. Accepted Fisher iterations, unsupported,
        constrained, SCOP, and cached-working-system routes retain Fisher scoring.
        Public callers should leave this True.
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

    structured_decision = resolve_structured_backend(
        gms,
        groups,
        direct_solve=direct_solve,
        coefficient_width=p,
    )
    _use_structured = structured_decision.use_structured
    _structured_group_index = structured_decision.group_index
    _direct_fallback_reason = structured_decision.fallback_reason
    _structured_layout = (
        get_scalar_structured_layout(
            dm,
            groups,
            dominant_group_index=_structured_group_index,
        )
        if _use_structured and _structured_group_index is not None
        else None
    )

    if offset is None:
        offset = np.zeros(n)

    beta = beta_init.copy() if beta_init is not None else np.zeros(p)

    if intercept_init is not None:
        intercept = intercept_init
    else:
        mu0 = initial_mean(y, weights, family)
        intercept = float(link.link(np.atleast_1d(mu0))[0])

    # Dense paths retain the existing p x p penalty oracle. Structured paths
    # add each penalty directly to A or d, unless a caller already supplied a
    # dense override (which remains authoritative).
    S: NDArray | None
    if _use_structured:
        S = None if S_override is None else np.asarray(S_override, dtype=np.float64)
        if S is None and reml_penalties is None:
            raise ValueError(
                "direct_solve='structured' requires compact reml_penalties "
                "when S_override is not supplied."
            )
    elif S_override is not None:
        S = S_override
    else:
        S = _build_penalty_matrix(gms, groups, lambda2, p, reml_penalties=reml_penalties)

    def penalty_matvec(beta_values: NDArray) -> NDArray:
        """Apply the fitted penalty without expanding an identity random-effect block."""
        values = np.asarray(beta_values, dtype=np.float64)
        if S is not None:
            return S @ values
        if reml_penalties is None:  # pragma: no cover - validated above
            raise RuntimeError("Structured penalty components are unavailable.")
        from superglm.reml.penalty_algebra import penalty_component_matvec

        product = np.zeros_like(values)
        for component in reml_penalties:
            lam = float(lambda2[component.name]) if isinstance(lambda2, dict) else float(lambda2)
            if lam == 0.0:
                continue
            product[component.group_sl] += lam * penalty_component_matvec(
                component,
                values[component.group_sl],
                gms[component.group_index],
            )
        return product

    def penalty_quadratic(beta_values: NDArray) -> float:
        values = np.asarray(beta_values, dtype=np.float64)
        return float(values @ penalty_matvec(values))

    trace_enabled = trace_run is not None and trace_run.enabled
    trace_basis_id = trace_run.next_basis_id() if trace_enabled and trace_run is not None else None
    if not trace_enabled:
        resolved_lambdas: tuple[tuple[str, object], ...] = ()
    elif isinstance(lambda2, dict):
        resolved_lambdas = tuple(
            (f"smooth:{name}", float(value)) for name, value in sorted(lambda2.items())
        )
    else:
        resolved_lambdas = (("smooth", float(lambda2)),)

    def emit_evaluation(
        state: _IRLSState,
        *,
        phase: str,
        iteration: int,
        alpha: float | None = None,
        enclosing_proposal_state_id: int | None = None,
        deviance_reused: bool = False,
    ) -> None:
        if not trace_enabled:
            return
        assert trace_run is not None
        trace_run.emit_lazy(
            "evaluation",
            lambda: {
                "state_id": state.state_id,
                "evaluation_id": state.evaluation_id,
                "solver": "irls_direct",
                "phase": phase,
                "outer_iteration": iteration,
                "trial_alpha": alpha,
                "enclosing_proposal_state_id": enclosing_proposal_state_id,
                "state_space": state.state_space,
                "basis_id": state.basis_id,
                "lambdas": state.lambdas,
                "dispersion": state.dispersion,
                "intercept": state.intercept,
                "deviance": state.deviance,
                "penalized_deviance": state.penalized_deviance,
                "deviance_source": "provided" if deviance_reused else "evaluated",
            },
            channel="pirls",
            purpose=trace_purpose,
            authoritative=False,
        )

    def evaluate_state(
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
    ) -> _IRLSState:
        if trace_enabled:
            assert trace_run is not None
            state_id = trace_run.next_state_id()
            evaluation_id = trace_run.next_evaluation_id()
        else:
            state_id = None
            evaluation_id = None
        state = _evaluate_irls_state(
            dm,
            y,
            weights,
            family,
            link,
            offset,
            beta_values,
            intercept_value,
            deviance=deviance,
            eta_unclipped=eta_unclipped,
            state_id=state_id,
            evaluation_id=evaluation_id,
            basis_id=trace_basis_id,
            lambdas=resolved_lambdas,
        )
        if not _has_scop:
            state = replace(
                state,
                penalized_deviance=float(state.deviance + penalty_quadratic(state.beta)),
            )
        if emit_trace:
            emit_evaluation(
                state,
                phase=phase,
                iteration=iteration,
                alpha=alpha,
                enclosing_proposal_state_id=enclosing_proposal_state_id,
                deviance_reused=deviance is not None,
            )
        return state

    def emit_state_commit(
        state: _IRLSState,
        *,
        iteration: int,
        fit_converged: bool,
        convergence_value: float | None,
        termination_reason: str | None,
    ) -> None:
        if not trace_enabled:
            return
        assert trace_run is not None
        trace_run.emit_lazy(
            "state_commit",
            lambda: {
                "state_id": state.state_id,
                "evaluation_id": state.evaluation_id,
                "solver": "irls_direct",
                "phase": "initial" if iteration == 0 else "outer",
                "outer_iteration": iteration,
                "state_space": state.state_space,
                "basis_id": state.basis_id,
                "lambdas": state.lambdas,
                "dispersion": state.dispersion,
                "intercept": state.intercept,
                "deviance": state.deviance,
                "penalized_deviance": state.penalized_deviance,
                "fit_converged": fit_converged,
                "convergence_criterion": convergence,
                "convergence_value": convergence_value,
                "convergence_tolerance": tol,
                "termination_reason": termination_reason,
            },
            channel="pirls",
            purpose=trace_purpose,
        )

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
    _scop_curvature = "fisher"
    if _has_scop:
        from superglm.reml.observed_geometry import classify_scop_reml_curvature

        _scop_curvature = classify_scop_reml_curvature(family, link)
    if (not _compute_fit_statistics and compute_rank_info) or (
        _return_working_system and (compute_rank_info or has_constraints or _has_scop)
    ):
        raise ValueError("intermediate REML shortcuts require rank metadata to be disabled")
    if not _compute_reml_geometry and (compute_rank_info or _compute_fit_statistics):
        raise ValueError(
            "omitting generic REML geometry requires rank metadata and fit statistics "
            "to be disabled"
        )
    _observed_newton_available = bool(
        _use_observed_newton
        and not has_constraints
        and not _has_scop
        and not _return_working_system
        and supports_observed_newton(family, link)
    )
    _observed_newton_active = False
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
        _reduced_gms = []
        for gi in _non_scop_groups_idx:
            _reduced_gms.append(gms[gi])
        _reduced_dm = DesignMatrix(_reduced_gms, n=n, p=_p_reduced)
        _reduced_tabmat_state = TabmatCenteringState()

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
        provisional = evaluate_state(
            beta,
            intercept,
            phase="scop_initialization",
            iteration=0,
        )
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

    # Tabmat acceleration: the structured layout owns a pruned small-block
    # plan, so it must not construct a split containing the dominant factor.
    # Other non-discrete paths retain the shared full-design split behavior.
    if _use_structured:
        _tabmat_split = None
    elif _use_qr:
        _tabmat_split = None
    elif has_constraints:
        # The constrained raw-moment path uses the cached execution plan.
        _tabmat_split = None
    else:
        # Ordinary intercept profiling currently benefits only when the split
        # contains a native high-cardinality categorical component. Avoid
        # materializing an unused dense duplicate for numeric and low-cardinality fits.
        _tabmat_split = dm.tabmat_centering_split
    _can_reuse_weighted_gram = _has_constant_irls_weights(family, link) and not _has_scop
    _can_reuse_weighted_gram = _can_reuse_weighted_gram and not _use_structured
    dm.execution_plan.validate_group_spans(groups)
    _defer_raw_spline = (
        not dm.raw_spline_tabmat_plan_built
        and _is_raw_spline_tabmat_centering_candidate(gms, n=n)
        and (
            _defer_raw_spline_tabmat_plan(
                n=n,
                raw_width=sum(int(group.B.shape[1]) for group in gms if group.shape[1] > 0),
                constant_weights=_can_reuse_weighted_gram,
                repeated_fit=trace_purpose
                in {"reml_bootstrap", "reml_candidate", "reml_line_search"},
            )
        )
    )
    _tabmat_centering_state = TabmatCenteringState(
        raw_spline_eligible=False if _defer_raw_spline else None
    )
    if profile is not None and _defer_raw_spline:
        profile["centered_spline_tabmat_cold_policy_rejections"] = (
            profile.get("centered_spline_tabmat_cold_policy_rejections", 0) + 1
        )
    _constant_w_gram_cache: tuple[NDArray, NDArray, float] | None = None
    _constant_centered_cache: CenteredSystem | None = None
    _constant_centered_z: NDArray | None = None
    _centered_factor_certification: _CenteredFactorCertification | None = None

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
            penalty=np.asarray(S),
            tabmat_split=_tabmat_split,
            tabmat_state=_tabmat_centering_state,
            profile=profile,
        )
        if _can_reuse_weighted_gram:
            _constant_centered_cache = system
            _constant_centered_z = z_off_current.copy()
        return system

    def certify_centered_factor(
        system: CenteredSystem,
        W_current: NDArray,
        *,
        response: NDArray | None = None,
    ) -> _CenteredFactorCertification:
        """Return a factor certificate for one immutable centered geometry."""
        nonlocal _centered_factor_certification
        cached = _centered_factor_certification
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
            and (
                response is None or (cached.system is system and cached.transformed_rhs is not None)
            )
        ):
            return cached

        if response is None:
            factor = grouped_augmented_factor(
                dm,
                W_current,
                system.penalty,
                center=system.mean_x,
            )
            transformed_rhs = None
        else:
            factor, transformed_rhs = grouped_augmented_factor_rhs(
                dm,
                W_current,
                system.penalty,
                response=response,
                center=system.mean_x,
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
        _centered_factor_certification = cached
        return cached

    t_start = time.perf_counter()
    converged = False
    XtWX_beta: NDArray | SymmetricBlockOperator | None = None
    _final_structured_system: ScalarStructuredSystem | None = None
    _final_penalized_operator: SymmetricBlockOperator | None = None

    # Phase timing accumulators
    _t_working = 0.0
    _t_gram = 0.0
    _t_solve = 0.0
    _t_deviance = 0.0
    _t_eta = 0.0
    _t_deviance_eval = 0.0
    _last_working_centered: CenteredSystem | None = None
    _last_working_structured: ScalarStructuredSystem | None = None

    # Freeze the fit-entry state so iteration-one trial safety has a baseline.
    committed = evaluate_state(
        beta,
        intercept,
        phase="initial",
        iteration=0,
        deviance=_deviance_init,
        emit_trace=not _has_scop,
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

        def with_scop_merit(trial: _SCOPTrialState) -> _SCOPTrialState:
            """Attach deviance plus the latent-coordinate quadratic penalty."""
            penalty_quad = float(trial.irls.beta @ S @ trial.irls.beta)
            for group_state in trial.groups:
                group = groups[group_state.group_index]
                group_slice = group.sl
                block = S[group_slice, group_slice]
                penalty_quad -= float(group_state.gamma_eff @ block @ group_state.gamma_eff)
                lam_scop = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
                latent_penalty = _scop_specs[group_state.group_index].S_scop
                penalty_quad += float(
                    lam_scop * (group_state.beta_eff @ latent_penalty @ group_state.beta_eff)
                )
            return replace(
                trial,
                irls=replace(
                    trial.irls,
                    penalized_deviance=float(trial.irls.deviance + penalty_quad),
                ),
            )

        scop_committed = with_scop_merit(scop_committed)
        committed = scop_committed.irls
        emit_evaluation(
            committed,
            phase="initial",
            iteration=0,
            deviance_reused=_deviance_init is not None,
        )
    emit_state_commit(
        committed,
        iteration=0,
        fit_converged=False,
        convergence_value=None,
        termination_reason=None,
    )
    objective_prev = (
        committed.deviance if committed.penalized_deviance is None else committed.penalized_deviance
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

    max_halving = 20  # max step-halving attempts per iteration
    _consecutive_svd = 0  # for auto-mode warning

    for it in range(max_iter):
        beta_prev = committed.beta
        intercept_prev = committed.intercept
        beta = committed.beta.copy()
        intercept = committed.intercept
        committed_active_set = None if prev_active_set is None else list(prev_active_set)
        rank_truncated: bool | None = None
        used_rank_certification = False
        scop_proposal_eta_unclipped: NDArray | None = None

        # Working quantities from current eta/mu (already computed)
        _t0 = time.perf_counter()
        working_rows = coefficient_working_rows(
            distribution=family,
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=weights,
            prefer_observed=_observed_newton_active,
        )
        W = working_rows.weights
        z = working_rows.response
        if working_rows.fallback_reason is not None:
            # Once exact observed rows fail their fit-wide safety contract,
            # keep all later proposals on one coherent Fisher-scoring route.
            _observed_newton_active = False
            _observed_newton_available = False
            _can_reuse_weighted_gram = (
                _has_constant_irls_weights(family, link) and not _use_structured
            )
            _constant_w_gram_cache = None
            _constant_centered_cache = None
            _constant_centered_z = None
            if profile is not None:
                profile["irls_observed_newton_fallbacks"] = (
                    profile.get("irls_observed_newton_fallbacks", 0) + 1
                )
        elif working_rows.curvature_source == "observed" and profile is not None:
            profile["irls_observed_newton_iters"] = profile.get("irls_observed_newton_iters", 0) + 1
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
            iteration_rank = decompose_factor(A, retain_factor_solve=True)
            beta = iteration_rank.solve_factor_rhs(rhs_qr)
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

                # Step 3: Profile the intercept in stable centered
                # coordinates.  The raw augmented Gram loses the ordinary
                # slope entirely after a large column translation.
                reduced_centered = build_centered_system(
                    dm=_reduced_dm,
                    W=W,
                    z_off=z_adj,
                    penalty=_S_reduced,
                    tabmat_split=_reduced_dm.tabmat_centering_split,
                    tabmat_state=_reduced_tabmat_state,
                    profile=profile,
                )
                reduced_scale = np.sqrt(
                    np.maximum(np.diag(reduced_centered.data_gram), 0.0) / reduced_centered.sum_w
                )
                use_anchor_centering = not _raw_centering_well_scaled(
                    reduced_centered.mean_x,
                    reduced_scale,
                )
                if use_anchor_centering:
                    reduced_centered = build_anchor_centered_system(
                        dm=_reduced_dm,
                        W=W,
                        z_off=z_adj,
                        penalty=_S_reduced,
                    )
                _t_gram += time.perf_counter() - _t0

                # Step 4: Solve for unconstrained coefficients
                _t0 = time.perf_counter()
                reduced_rank = decompose_gram(reduced_centered.hessian)
                reduced_factor_rhs = None
                if needs_factor_certification(reduced_rank):
                    reduced_factor, certified_rhs = grouped_augmented_factor_rhs(
                        _reduced_dm,
                        W,
                        reduced_centered.penalty,
                        response=z_adj - reduced_centered.mean_z,
                        center=reduced_centered.mean_x,
                    )
                    certified = decompose_factor(
                        reduced_factor,
                        retain_factor_solve=True,
                    )
                    reduced_rank = certified
                    reduced_factor_rhs = certified_rhs
                    used_rank_certification = True
                beta_reduced = (
                    reduced_rank.solve(reduced_centered.rhs)
                    if reduced_factor_rhs is None
                    else reduced_rank.solve_factor_rhs(reduced_factor_rhs)
                )
                intercept = reduced_centered.mean_z - float(reduced_centered.mean_x @ beta_reduced)
                _cond_est = reduced_rank.pre_truncation_condition
                _used_svd = reduced_rank.used_svd_fallback
                rank_truncated = reduced_rank.rank_truncated

                # Scatter reduced beta back into full beta vector
                beta = np.zeros(p)
                beta[_reduced_to_full] = beta_reduced
                _t_solve += time.perf_counter() - _t0

                # Step 5: Compute residual for SCOP Newton step
                if use_anchor_centering:
                    eta_unconstrained = reduced_centered.mean_z + stable_centered_matvec(
                        dm=_reduced_dm,
                        beta=beta_reduced,
                        W=W,
                        sum_w=reduced_centered.sum_w,
                    )
                else:
                    eta_unconstrained = intercept + _reduced_dm.matvec(beta_reduced)

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
                scop_proposal_eta_unclipped = eta_unconstrained + offset
                for gi, st in _scop_state.items():
                    gamma_eff = st["reparam"].forward(scop_results[gi].beta_new)
                    eta_group = st["B_scop"] @ gamma_eff
                    if st["bin_idx"] is not None:
                        eta_group = eta_group[st["bin_idx"]]
                    scop_proposal_eta_unclipped += eta_group

            elif _use_structured:
                if _structured_group_index is None:  # pragma: no cover - selection invariant
                    raise RuntimeError("Structured backend has no dominant group.")
                Wz = W * z_off
                structured_system = build_scalar_structured_system(
                    gms,
                    groups,
                    W,
                    Wz,
                    dominant_group_index=_structured_group_index,
                    tabmat_split=_tabmat_split,
                    layout=_structured_layout,
                )
                penalized_operator = build_penalized_scalar_operator(
                    structured_system,
                    gms,
                    groups,
                    lambda2,
                    reml_penalties=reml_penalties,
                    S_override=S_override,
                )
                _last_working_structured = structured_system
                _final_structured_system = structured_system
                _final_penalized_operator = penalized_operator
                _t_gram += time.perf_counter() - _t0

                _t0 = time.perf_counter()
                augmented_factor, rhs = build_augmented_scalar_factor(
                    structured_system,
                    penalized_operator,
                )
                beta_aug = augmented_factor.solve(rhs)
                intercept = float(beta_aug[0])
                beta = beta_aug[1:]
                _used_svd = augmented_factor.used_dense_fallback
                _cond_est = augmented_factor.schur_condition_estimate
                rank_truncated = augmented_factor.rank_truncated
                _t_solve += time.perf_counter() - _t0
            elif not has_constraints:
                centered = get_centered_system(W, z_off)
                _last_working_centered = centered
                _t_gram += time.perf_counter() - _t0
                _t0 = time.perf_counter()
                iteration_rank = decompose_gram(centered.hessian)
                iteration_factor_rhs = None
                if needs_factor_certification(iteration_rank):
                    certification = certify_centered_factor(
                        centered,
                        W,
                        response=z_off - centered.mean_z,
                    )
                    certified = certification.decomposition
                    if certification.transformed_rhs is None:  # pragma: no cover - invariant
                        raise RuntimeError("factor certification omitted its transformed RHS")
                    iteration_rank = certified
                    iteration_factor_rhs = certification.transformed_rhs
                    used_rank_certification = True
                beta = (
                    iteration_rank.solve(centered.rhs)
                    if iteration_factor_rhs is None
                    else iteration_rank.solve_factor_rhs(iteration_factor_rhs)
                )
                intercept = centered.mean_z - float(centered.mean_x @ beta)
                _cond_est = iteration_rank.pre_truncation_condition
                _used_svd = iteration_rank.used_svd_fallback
                rank_truncated = iteration_rank.rank_truncated
                _t_solve += time.perf_counter() - _t0
            else:
                Wz = W * z_off
                sum_W, sum_Wz = _working_sums(W, Wz)

                if _can_reuse_weighted_gram and _constant_w_gram_cache is not None:
                    XtWX, XtW1, sum_W = _constant_w_gram_cache
                    XtWz = dm.rmatvec(Wz)
                else:
                    # Combined gram + rmatvec: shares O(n) bincount for discretized groups
                    moments = dm.execution_plan._moments_prevalidated(
                        W,
                        rhs=(Wz,),
                        include_xtw=True,
                        profile=profile,
                    )
                    if moments.xtw is None:  # pragma: no cover - requested above
                        raise RuntimeError("execution plan did not return X'W")
                    XtWX = moments.gram
                    XtW1 = moments.xtw
                    XtWz = moments.xt_rhs[0]
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
                rhs[0] = sum_Wz
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

            # A bounded factor pass can intentionally replace an uncertain Gram
            # rank.  Preserve that in iteration diagnostics, but do not report it
            # as a failed solve or recommend switching the whole fit to dense QR.
            warnable_svd_fallback = _used_svd and not used_rank_certification
            if warnable_svd_fallback:
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
            proposal_irls = evaluate_state(
                beta,
                intercept,
                phase="scop_proposal",
                iteration=it + 1,
                alpha=1.0,
                eta_unclipped=scop_proposal_eta_unclipped,
                emit_trace=False,
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
            proposal_scop = with_scop_merit(proposal_scop)
            proposal_irls = proposal_scop.irls
            emit_evaluation(
                proposal_irls,
                phase="scop_proposal",
                iteration=it + 1,
                alpha=1.0,
            )
            scop_trial_cache: dict[float, _SCOPTrialState] = {1.0: proposal_scop}

            def evaluate_scop_trial(alpha: float) -> _IRLSState:
                if trace_enabled:
                    assert trace_run is not None
                    state_id = trace_run.next_state_id()
                    evaluation_id = trace_run.next_evaluation_id()
                else:
                    state_id = None
                    evaluation_id = None
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
                    state_id=state_id,
                    evaluation_id=evaluation_id,
                    basis_id=trace_basis_id,
                    lambdas=resolved_lambdas,
                )
                candidate = with_scop_merit(candidate)
                emit_evaluation(
                    candidate.irls,
                    phase="scop_line_search_trial",
                    iteration=it + 1,
                    alpha=alpha,
                    enclosing_proposal_state_id=proposal_scop.irls.state_id,
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
            proposal = evaluate_state(
                beta,
                intercept,
                phase="proposal",
                iteration=it + 1,
                alpha=1.0,
            )
            trial_cache: dict[float, _IRLSState] = {1.0: proposal}
            trial_directions: tuple[NDArray, float, NDArray] | None = None

            def evaluate_trial(alpha: float) -> _IRLSState:
                nonlocal trial_directions
                if trial_directions is None:
                    trial_directions = (
                        proposal.beta - committed.beta,
                        proposal.intercept - committed.intercept,
                        proposal.eta_unclipped - committed.eta_unclipped,
                    )
                beta_direction, intercept_direction, eta_direction = trial_directions
                beta_trial = committed.beta + alpha * beta_direction
                intercept_trial = committed.intercept + alpha * intercept_direction
                eta_trial = committed.eta_unclipped + alpha * eta_direction
                candidate = evaluate_state(
                    beta_trial,
                    intercept_trial,
                    phase="line_search_trial",
                    iteration=it + 1,
                    alpha=alpha,
                    eta_unclipped=eta_trial,
                    enclosing_proposal_state_id=proposal.state_id,
                )
                trial_cache[alpha] = candidate
                return candidate

            decision = _select_irls_trial(
                committed=committed,
                proposal=proposal,
                evaluate_state=evaluate_trial,
                max_halving=max_halving,
                merit_delta=lambda candidate, base: _stable_penalized_deviance_delta(
                    candidate,
                    base,
                    penalty_matvec,
                ),
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

        proposal_state = proposal_scop.irls if _has_scop else proposal
        dev_rel_change = None
        coef_change = None
        if np.isfinite(dev):
            if convergence == "coefficients":
                coef_change = float(
                    np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta)))
                )
                coef_change = max(
                    coef_change,
                    abs(intercept - intercept_prev) / max(1.0, abs(intercept)),
                )
                if _has_scop:
                    latent_change = max(
                        float(
                            np.max(
                                np.abs(retained_group.beta_eff - committed_group.beta_eff)
                                / np.maximum(1.0, np.abs(retained_group.beta_eff))
                            )
                        )
                        for retained_group, committed_group in zip(
                            retained_scop.groups,
                            scop_committed.groups,
                            strict=True,
                        )
                    )
                    coef_change = max(coef_change, latent_change)
                converged_this_iter = coef_change < tol
                convergence_value = coef_change
            else:
                objective = (
                    retained.deviance
                    if retained.penalized_deviance is None
                    else retained.penalized_deviance
                )
                if np.isfinite(objective_prev):
                    dev_rel_change = abs(objective - objective_prev) / (abs(objective_prev) + 1.0)
                converged_this_iter = dev_rel_change is not None and dev_rel_change < tol
                convergence_value = dev_rel_change
        else:
            converged_this_iter = False
            convergence_value = None
        if step_rejected:
            converged_this_iter = False

        curvature_rescue_activated = bool(
            step_rejected
            and _observed_newton_available
            and not _observed_newton_active
            and it + 1 < max_iter
        )
        fisher_fallback_activated = bool(
            step_rejected and _observed_newton_active and it + 1 < max_iter
        )

        if fisher_fallback_activated:
            termination_reason = "curvature_fallback"
        elif curvature_rescue_activated:
            termination_reason = "curvature_rescue"
        elif step_rejected:
            termination_reason = "step_rejected"
        elif not np.isfinite(dev):
            termination_reason = "nonfinite_deviance"
        elif converged_this_iter:
            termination_reason = "converged"
        elif it + 1 == max_iter:
            termination_reason = "max_iter"
        else:
            termination_reason = "continue"

        if trace_enabled:
            assert trace_run is not None
            trace_run.emit_lazy(
                "step_decision",
                lambda: {
                    "solver": "irls_direct",
                    "outer_iteration": it + 1,
                    "base_state_id": committed.state_id,
                    "proposal_state_id": proposal_state.state_id,
                    "committed_state_id": retained.state_id,
                    "accepted_alpha": decision.alpha,
                    "step_halvings": decision.step_halvings,
                    "trials_attempted": decision.trials_attempted,
                    "step_rejected": decision.step_rejected,
                    "fit_converged": converged_this_iter,
                    "convergence_criterion": convergence,
                    "convergence_value": convergence_value,
                    "convergence_tolerance": tol,
                    "termination_reason": termination_reason,
                    "working_curvature": working_rows.curvature_source,
                    "curvature_rescue_activated": curvature_rescue_activated,
                    "curvature_fallback_activated": fisher_fallback_activated,
                },
                channel="pirls",
                purpose=trace_purpose,
            )
        if np.isfinite(dev):
            emit_state_commit(
                retained,
                iteration=it + 1,
                fit_converged=converged_this_iter,
                convergence_value=convergence_value,
                termination_reason=termination_reason,
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
                    trials_attempted=decision.trials_attempted,
                    accepted_alpha=decision.alpha,
                    base_state_id=committed.state_id,
                    proposal_state_id=proposal_state.state_id,
                    committed_state_id=retained.state_id,
                    evaluation_id=retained.evaluation_id,
                    state_space=retained.state_space,
                    basis_id=retained.basis_id,
                    convergence_criterion=convergence,
                    convergence_value=convergence_value,
                    convergence_tolerance=tol,
                    termination_reason=termination_reason,
                )
            )

        logger.info(
            f"  irls_direct iter={it + 1:3d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}"
        )

        if not np.isfinite(dev):
            logger.warning(f"IRLS direct non-finite deviance at iter={it + 1}: dev={dev:.2e}")
            break

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
                    "trials_attempted": int(decision.trials_attempted),
                    "step_rejected": bool(step_rejected),
                    "base_state_id": committed.state_id,
                    "proposal_state_id": proposal_state.state_id,
                    "committed_state_id": retained.state_id,
                    "rank_truncated": rank_truncated,
                    "cond_estimate": float(_cond_est),
                    "used_svd_fallback": bool(_used_svd),
                    "has_scop": bool(_has_scop),
                    "working_curvature": working_rows.curvature_source,
                    "curvature_rescue_activated": bool(curvature_rescue_activated),
                    "curvature_fallback_activated": bool(fisher_fallback_activated),
                },
            )

        if step_rejected and not (curvature_rescue_activated or fisher_fallback_activated):
            logger.warning(
                "IRLS direct rejected all trial steps at iter=%d; restored committed state",
                it + 1,
            )
            break

        if curvature_rescue_activated:
            _observed_newton_active = True
            _can_reuse_weighted_gram = False
            _constant_w_gram_cache = None
            _constant_centered_cache = None
            _constant_centered_z = None
            if profile is not None:
                profile["irls_observed_newton_rescues"] = (
                    profile.get("irls_observed_newton_rescues", 0) + 1
                )
            logger.info(
                "IRLS direct switching Gamma/log coefficient proposals to observed "
                "Newton curvature after iteration %d",
                it + 1,
            )
        elif fisher_fallback_activated:
            _observed_newton_active = False
            _observed_newton_available = False
            _can_reuse_weighted_gram = _has_constant_irls_weights(family, link)
            _constant_w_gram_cache = None
            _constant_centered_cache = None
            _constant_centered_z = None
            if profile is not None:
                profile["irls_observed_newton_rejections"] = (
                    profile.get("irls_observed_newton_rejections", 0) + 1
                )
            logger.info(
                "IRLS direct rejected an observed Gamma/log proposal at iteration %d; "
                "restoring Fisher scoring",
                it + 1,
            )

        committed = retained
        if _has_scop:
            scop_committed = retained_scop
        if converged_this_iter:
            converged = True
            break
        dev_prev = dev
        objective_prev = (
            retained.deviance
            if retained.penalized_deviance is None
            else retained.penalized_deviance
        )

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
            # Recover the retained non-SCOP contribution from the exact
            # committed predictor. Rebuilding X beta + intercept would lose
            # several score digits for a translated ordinary column.
            eta_unconstrained = eta_unclipped - offset
            for st in _scop_state.values():
                eta_group = st["B_scop"] @ st["gamma_eff"]
                if st["bin_idx"] is not None:
                    eta_group = eta_group[st["bin_idx"]]
                eta_unconstrained = eta_unconstrained - eta_group
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
                    _scop_state[gi]["last_fisher_fallback"] = bool(refresh.used_fisher_fallback)
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
                    st["last_fisher_fallback"] = bool(refresh.used_fisher_fallback)

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
    # the centered inference system. The structured path retains the same
    # moments in block form and never materializes the dominant K x K block.
    centered_final: CenteredSystem | None = None
    structured_final: ScalarStructuredSystem | None = None
    if _use_structured:
        if _return_working_system:
            if _last_working_structured is None:
                raise RuntimeError("working structured system was not computed")
            structured_final = _last_working_structured
        else:
            if _structured_group_index is None:  # pragma: no cover - selection invariant
                raise RuntimeError("Structured backend has no dominant group.")
            z_off = z - offset
            structured_final = build_scalar_structured_system(
                gms,
                groups,
                W,
                W * z_off,
                dominant_group_index=_structured_group_index,
                tabmat_split=_tabmat_split,
                layout=_structured_layout,
            )
        _final_structured_system = structured_final
        _final_penalized_operator = build_penalized_scalar_operator(
            structured_final,
            gms,
            groups,
            lambda2,
            reml_penalties=reml_penalties,
            S_override=S_override,
        )
        XtW1 = np.empty(p, dtype=np.float64)
        XtW1[structured_final.operator.small_indices] = structured_final.xtw_small
        XtW1[structured_final.operator.structured_indices] = structured_final.xtw_structured
        XtWz = np.empty(p, dtype=np.float64)
        XtWz[structured_final.operator.small_indices] = structured_final.xtwz_small
        XtWz[structured_final.operator.structured_indices] = structured_final.xtwz_structured
        sum_W = structured_final.sum_w
        sum_Wz = structured_final.sum_wz
        mean_x = XtW1 / sum_W
        mean_z = sum_Wz / sum_W
        centered_rhs = XtWz - XtW1 * mean_z
        XtWX = None
    else:
        if _return_working_system:
            if _last_working_centered is None:
                raise RuntimeError("working centered system was not computed")
            centered_final = _last_working_centered
        else:
            z_off = z - offset
            centered_final = get_centered_system(W, z_off)
        XtWX, XtW1, XtWz, sum_Wz = centered_final.raw_weighted_moments()
        sum_W = centered_final.sum_w
        mean_x = centered_final.mean_x
        mean_z = centered_final.mean_z
        centered_rhs = centered_final.rhs

    # Cache final-iteration raw and stable centered quantities for the cached-W
    # fREML optimizer. These allow re-solving the profiled-intercept system with
    # a new penalty matrix S without any data passes (O(p³), not O(n·K²)).
    if cache_out is not None:
        if _use_structured:
            if structured_final is None:  # pragma: no cover - branch invariant
                raise RuntimeError("Structured fit did not produce final sufficient statistics.")
            cache_out["structured_system"] = structured_final
            cache_out["structured_operator"] = structured_final.operator
            cache_out["penalized_operator"] = _final_penalized_operator
            cache_out["xtwz_small"] = structured_final.xtwz_small
            cache_out["xtwz_structured"] = structured_final.xtwz_structured
        else:
            if centered_final is None or XtWX is None:  # pragma: no cover - branch invariant
                raise RuntimeError("Dense fit did not produce a centered system.")
            cache_out["XtWX"] = XtWX
            cache_out["centered_XtWX"] = centered_final.data_gram
        cache_out["XtWz"] = XtWz
        cache_out["XtW1"] = XtW1
        cache_out["sum_W"] = sum_W
        cache_out["sum_Wz"] = sum_Wz
        cache_out["centered_rhs"] = centered_rhs
        cache_out["mean_x"] = mean_x
        cache_out["mean_z"] = mean_z
        if _has_scop:
            # The SCOP LAML mode certificate must evaluate the exact retained
            # predictor. Reconstructing a huge translated column plus its
            # compensating intercept can otherwise manufacture a false KKT
            # residual several orders above tolerance.
            cache_out["eta_unclipped"] = eta_unclipped

    # REML works in the full (intercept, slopes) coefficient space.  Profiling
    # the unpenalized intercept yields the centered Schur complement H_c.  Its
    # inverse is the slope block of H_aug^{-1}, while the unit-determinant
    # centering transform gives log|H_aug| = log(sum(W)) + log|H_c| at full
    # rank. With aliases, the same expression is the retained centered-space
    # determinant measure, not the raw augmented pseudo-determinant.
    _t0 = time.perf_counter()
    structured_factor: ProfiledScalarSchurFactor | None = None
    if _use_structured:
        if structured_final is None or _final_penalized_operator is None:
            raise RuntimeError("Structured fit did not produce final coefficient blocks.")
        augmented_factor, _ = build_augmented_scalar_factor(
            structured_final,
            _final_penalized_operator,
        )
        structured_factor = ProfiledScalarSchurFactor(
            augmented_factor=augmented_factor,
            sum_w=structured_final.sum_w,
            xtw=XtW1,
        )
        XtWX_beta = structured_final.operator
        if _compute_reml_geometry:
            XtWX_S_inv_beta: NDArray | HessianFactor = structured_factor
            log_det_H: float | None = augmented_factor.logdet()
            reml_hessian_rank: int | None = augmented_factor.rank
        else:
            XtWX_S_inv_beta = np.empty((0, 0), dtype=np.float64)
            log_det_H = None
            reml_hessian_rank = None
        if _compute_fit_statistics:
            p_eff = 1.0 + structured_factor.trace_inverse_operator(XtWX_beta)
        else:
            p_eff = 0.0
        # Structured retained-fit inference consumes the factor directly. A
        # dense RankInfo would defeat the backend's O(K q + q²) memory bound.
        rank_info = None
    else:
        if centered_final is None or XtWX is None:  # pragma: no cover - branch invariant
            raise RuntimeError("Dense fit did not produce a centered system.")
        XtWX_beta = XtWX
        reml_slope_rank: RankDecomposition | None
        if _compute_reml_geometry:
            reml_slope_rank = decompose_gram(centered_final.hessian)
            if needs_factor_certification(reml_slope_rank):
                certification = certify_centered_factor(
                    centered_final,
                    W,
                )
                reml_slope_rank = certification.decomposition
            XtWX_S_inv_beta = reml_slope_rank.pseudo_inverse()
            log_det_H = float(np.log(centered_final.sum_w) + reml_slope_rank.log_pdet)
            reml_hessian_rank = 1 + reml_slope_rank.rank
        else:
            reml_slope_rank = None
            XtWX_S_inv_beta = np.empty((0, 0), dtype=np.float64)
            log_det_H = None
            reml_hessian_rank = None

        coefficient_rank = None
        if _compute_fit_statistics and compute_rank_info:
            M_beta = centered_final.hessian + centered_final.sum_w * np.outer(
                centered_final.mean_x, centered_final.mean_x
            )
            coefficient_rank = decompose_gram(M_beta)
            if needs_factor_certification(coefficient_rank):
                certification = certify_centered_factor(centered_final, W)
                raw_factor = np.vstack(
                    (
                        certification.factor,
                        np.sqrt(centered_final.sum_w) * centered_final.mean_x,
                    )
                )
                coefficient_rank = decompose_factor(raw_factor)
        if _compute_fit_statistics:
            if reml_slope_rank is None:  # pragma: no cover - validated above
                raise RuntimeError("fit statistics require generic REML geometry")
            if _use_qr:
                sqrtW = np.sqrt(W)
                A_data_final = sqrtW[:, None] * (_X_full - centered_final.mean_x)
                data_rank = decompose_factor(A_data_final) if compute_rank_info else None
                augmented_rank = reml_slope_rank
            else:
                data_rank = decompose_gram(centered_final.data_gram) if compute_rank_info else None
                if data_rank is not None and needs_factor_certification(data_rank):
                    if not np.any(centered_final.penalty):
                        certification = certify_centered_factor(centered_final, W)
                        data_rank = certification.decomposition
                    else:
                        data_rank = decompose_factor(
                            grouped_weighted_factor(
                                dm,
                                W,
                                center=centered_final.mean_x,
                            )
                        )
                augmented_rank = reml_slope_rank
            feature_edf = np.diag(augmented_rank.pseudo_inverse() @ centered_final.data_gram).copy()
            feature_edf[np.abs(feature_edf) < 100.0 * np.finfo(float).eps] = 0.0
            p_eff = 1.0 + float(np.sum(feature_edf))
            if compute_rank_info:
                if data_rank is None:
                    raise RuntimeError("data-rank metadata was not computed")
                if coefficient_rank is None:
                    raise RuntimeError("coefficient-rank metadata was not computed")
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

    _resolved_direct_backend = "structured" if _use_structured else ("qr" if _use_qr else "gram")
    if profile is not None:
        profile["direct_backend"] = _resolved_direct_backend
        profile["direct_fallback_reason"] = _direct_fallback_reason
        if structured_factor is not None:
            profile["structured_dominant_group"] = structured_factor.dominant_group_name
            profile["structured_minimum_local_diagonal"] = structured_factor.minimum_local_diagonal
            profile["structured_schur_condition"] = structured_factor.schur_condition_estimate
            profile["structured_used_dense_fallback"] = structured_factor.used_dense_fallback
            profile["structured_fallback_reason"] = structured_factor.fallback_reason

    # Pearson-based phi for estimated-scale families. Gaussian/Gamma weights
    # are frequency weights; Tweedie weights are EDM prior weights.
    if _compute_fit_statistics and not getattr(family, "scale_known", True):
        V_final = np.maximum(family.variance(mu), _VARIANCE_FLOOR)
        pearson_chi2 = float(np.sum(weights * (y - mu) ** 2 / V_final))
        df_resid = pearson_residual_degrees_of_freedom(family, weights, p_eff)
        phi = pearson_chi2 / df_resid
    else:
        phi = 1.0
    if not _compute_reml_geometry:
        # Private SCOP candidates have no retained-fit statistics. NaN makes
        # accidental publication fail visibly instead of presenting 0 EDF or
        # unit dispersion as if either had been evaluated.
        p_eff = float("nan")
        phi = float("nan")

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
        reml_hessian_rank=reml_hessian_rank,
        rank_info=rank_info,
        state_id=retained.state_id,
        evaluation_id=retained.evaluation_id,
        state_space=retained.state_space,
        basis_id=retained.basis_id,
        termination_reason=termination_reason,
        direct_backend=_resolved_direct_backend,
        direct_fallback_reason=_direct_fallback_reason,
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

    if (
        _has_scop
        and scop_converged is not None
        and _compute_fit_statistics
        and _compute_scop_postfit_inference
    ):
        import superglm.reml.scop_geometry as scop_geometry

        if _scop_curvature == "observed":
            joint_geometry = scop_geometry.build_observed_scop_joint_geometry(
                dm=dm,
                distribution=family,
                link=link,
                y=y,
                sample_weight=weights,
                offset_arr=offset,
                result=result,
                penalty=S,
                scop_states=scop_converged,
                fisher_XtWX=XtWX,
                fisher_XtW1=XtW1,
                fisher_sum_W=sum_W,
                centered_fisher_gram=centered_final.data_gram,
                fisher_mean_x=centered_final.mean_x,
                eta_unclipped=eta_unclipped,
            )
        else:
            joint_geometry = scop_geometry.build_cached_scop_joint_geometry(
                raw_fisher_gram=XtWX,
                fisher_xtw=XtW1,
                fisher_sum_w=sum_W,
                latent_penalty=S,
                scop_states=scop_converged,
                centered_fisher_gram=centered_final.data_gram,
                fisher_mean_x=centered_final.mean_x,
                dm=dm,
                fisher_weights=W,
            )
        inference = scop_geometry.install_scop_postfit_inference(
            result,
            raw_fisher_gram=XtWX,
            centered_fisher_gram=centered_final.data_gram,
            fisher_xtw=XtW1,
            fisher_mean_x=centered_final.mean_x,
            fisher_sum_w=sum_W,
            latent_penalty=S,
            scop_states=scop_converged,
            groups=groups,
            observed_geometry=joint_geometry,
            dm=dm,
            fisher_weights=W,
        )

        # Estimated dispersion must use the same terminal EDF that downstream
        # covariance and summaries expose.  Known-scale likelihoods retain
        # their defining phi=1 rather than profiling a Pearson scale.
        if not getattr(family, "scale_known", True):
            V_final = np.maximum(family.variance(mu), _VARIANCE_FLOOR)
            pearson_chi2 = float(np.sum(weights * (y - mu) ** 2 / V_final))
            result.phi = pearson_chi2 / pearson_residual_degrees_of_freedom(
                family,
                weights,
                inference.total_edf,
            )
    if _expose_exact_support_state:
        result.scop_states = scop_converged

    if return_xtwx:
        if return_scop_state and scop_converged is not None:
            return result, XtWX_S_inv_beta, XtWX_beta, scop_converged
        return result, XtWX_S_inv_beta, XtWX_beta

    if return_scop_state and scop_converged is not None:
        return result, XtWX_S_inv_beta, scop_converged
    return result, XtWX_S_inv_beta
