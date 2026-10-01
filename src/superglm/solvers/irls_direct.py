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
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

import superglm.solvers.scop_exact_support as scop_exact_support
from superglm._blas_threads import keep_narrow_cap
from superglm._fit_trace import TraceRun
from superglm._group_matrix._group_matrix_centered import (
    _raw_centering_well_scaled,
    stable_centered_matvec,
)
from superglm._group_matrix._group_matrix_tabmat import (
    _defer_raw_spline_tabmat_plan,
    _is_raw_spline_tabmat_centering_candidate,
)
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
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import IdentityLink, Link, LogitLink, LogLink, stabilize_eta
from superglm.solvers._structured.block_leaves import factor_smooth_prior_statistics
from superglm.solvers.centered_system import (
    CenteredSystem,
    TabmatCenteringState,
    _FisherDataReuse,
    _InitialDataReuse,
    build_anchor_centered_system,
    build_centered_system,
    grouped_augmented_factor,
    grouped_augmented_factor_rhs,
    grouped_weighted_factor,
    refresh_centered_rhs,
)
from superglm.solvers.constrained_qp import (
    QPResult,
    _feasibility_slack,
    _is_feasible,
    solve_constrained_qp,
)
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom
from superglm.solvers.hessian_factor import HessianFactor
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _irls_objective_relative_change,
    _irls_objective_scale,
    _IRLSState,
    _IRLSStepDecision,
    _poisson_sqrt_halving_budget,
    _select_irls_trial,
    _stable_penalized_deviance_delta,
    _state_is_finite,
    mean_space_boundary_rows,
    mean_space_violation,
)
from superglm.solvers.mode_score import (
    MODE_CERTIFICATION_BAR,
    MODE_RESOLVE_CAP,
    ModeResidual,
    centre_offset_mean,
    centred_data_score,
    centred_intercept_remainder,
    centred_matvec,
    offset_columns,
    penalized_mode_residual,
    prior_weighted_centre,
    stagnation_window,
)
from superglm.solvers.pirls import (
    IterationDiagnostics,
    PIRLSResult,
    REMLGeometrySummary,
    TerminationReason,
    _extreme_weight_indices,
    _positive_working_weight_stats,
)
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankInfo,
    decompose_factor,
    decompose_gram,
    decompose_gram_if_authoritative,
    decompose_symmetric,
)
from superglm.solvers.scop import SCOPSolverReparam
from superglm.solvers.scop_newton import (
    _positive_quadratic_roundoff,
    scop_joint_newton_step,
    scop_newton_step,
)
from superglm.solvers.structured import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    FactorSmoothLeafFactor,
    FactorSmoothLeafLayout,
    FactorSmoothLeafSystem,
    FactorSmoothPenalizedOperator,
    NestedDataOperator,
    NestedPenalizedOperator,
    NestedSchurFactor,
    NestedStructuredLayout,
    NestedStructuredSystem,
    ProfiledFactorSmoothLeafFactor,
    ProfiledNestedSchurFactor,
    ProfiledSumToZeroTreeFactor,
    SumToZeroBlockOperator,
    SumToZeroLeafSystem,
    SumToZeroPenalizedOperator,
    SumToZeroTreeFactor,
    build_augmented_structured_factor,
    build_penalized_structured_operator,
    build_structured_system,
    cancelled_column_row_norms,
    centred_data_operator,
    compact_operator_diagonal,
    get_structured_layout,
    nested_prior_statistics,
    record_auto_backend_decision,
    resolve_structured_backend,
    solve_augmented_normal_equations,
)
from superglm.solvers.working_rows import (
    coefficient_initial_intercept,
    coefficient_working_rows,
    fisher_working_weights,
    pearson_chi2,
    supports_observed_newton,
)
from superglm.types import GroupSlice, LinearConstraintSet, PenaltyComponent

logger = logging.getLogger(__name__)

_QP_FEASIBILITY_TOL = 1e-12


def _solve_constrained_qp_with_cold_retry(
    H: NDArray,
    g: NDArray,
    A: NDArray,
    b: NDArray,
    active_set_init: list[int] | None,
) -> QPResult:
    """Retry a failed warm active set without weakening the KKT contract.

    A warm active set is an acceleration hint, not part of the mathematical
    problem. Near an active boundary, platform-dependent roundoff can make the
    warm solve project a stationary candidate and correctly withhold its KKT
    certificate even though the identical problem converges from a cold active
    set. Retry only that failed warm case, and replace its best iterate only
    when the ordinary cold solve returns a complete certificate.
    """
    result = solve_constrained_qp(
        H,
        g,
        A,
        b,
        active_set_init=active_set_init,
    )
    if active_set_init and not result.converged:
        cold_result = solve_constrained_qp(
            H,
            g,
            A,
            b,
            active_set_init=None,
        )
        if cold_result.converged:
            return cold_result
    return result


@dataclass(frozen=True)
class _SCOPGroupSpec:
    """Static definition of one SCOP group, separate from trial state."""

    group_index: int
    group: GroupSlice
    reparam: SCOPSolverReparam
    B_scop: NDArray
    S_scop: NDArray
    bin_idx: NDArray | None


def _immutable_or_none(values: NDArray | None) -> NDArray | None:
    """Freeze an optional array, preserving ``None``."""
    return None if values is None else _immutable_array(values)


@dataclass(frozen=True)
class _SCOPGroupState:
    """Dynamic SCOP state committed atomically with an IRLS snapshot."""

    group_index: int
    beta_eff: NDArray
    gamma_eff: NDArray
    H_scop_penalized: NDArray | None
    last_step_norm: float
    last_fisher_fallback: bool
    discarded_directions: NDArray | None = None
    """This group's columns of the directions the last Newton step could not
    resolve. Mode certification measures stationarity on the complement."""


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
                discarded_directions=proposed_group.discarded_directions,
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


def _structured_score_centre(
    system,
    factor,
    dm: DesignMatrix,
    W: NDArray,
    state_center: NDArray | None,
    offset_mask: NDArray | None,
) -> tuple:
    """``(mean_x, sum_w, centred diagonal, weakly identified slopes, mean_x - c)`` of a structured solve.

    The centring the certificate's score is formed about (``mode_residual``):
    the working-weighted means of the system, its centred data diagonal
    (offset-free for a nested chain, ``compact_operator_diagonal``), the
    slopes the factor truncated as weakly identified (one-engine design §3.6
    step 5, §3.9), and the means' offset from the state's centre
    (``centre_offset_mean`` on the columns ``offset_mask`` marks; ``None``
    without a column whose centre lies beyond its spread).
    """
    operator = system.operator
    xtw = np.empty(operator.shape[0], dtype=np.float64)
    xtw[operator.small_indices] = system.xtw_small
    xtw[operator.structured_indices] = system.xtw_structured
    mean_x = xtw / system.sum_w
    # an fs or sz system centres its c0-shifted moments (design §3.2), so a
    # large-offset column's centred diagonal does not cancel to noise
    diagonal = compact_operator_diagonal(centred_data_operator(system))
    excluded = tuple(
        index - 1 for index in getattr(factor, "weakly_identified_coefficients", ()) if index > 0
    )
    offset_mean = (
        None
        if state_center is None or offset_mask is None or not np.any(offset_mask)
        else centre_offset_mean(dm, W, float(system.sum_w), state_center, mean_x, offset_mask)
    )
    return mean_x, float(system.sum_w), diagonal, excluded, offset_mean


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


class StructuredSolverError(np.linalg.LinAlgError):
    """The structured solver cannot proceed with this fit.

    Raised in place of a structured factor's ``np.linalg.LinAlgError``, whose
    message it carries: the term, the tree node or border column, and the
    certificate or bound that failed (a tree pivot within its certified
    uncertainty, material negative Schur curvature, a retained border block
    that is not positive definite, exact intercept aliasing, non-finite
    statistics, a solve or selected inverse that is not representable).  No
    other solver is tried, under any ``direct_solve``: for a model its data
    identify this should not happen, and ``direct_solve="gram"`` fits the
    model with the dense solver instead.
    """


@contextmanager
def _structured_solver_errors():
    """Raise a structured factor's ``np.linalg.LinAlgError`` as ``StructuredSolverError``.

    Wraps every structured factor operation of a fit: the build and ``solve``
    at each iterate, the terminal build, ``logdet`` and
    ``trace_inverse_operator``, and the discrete line search's cached solve.
    The factors refuse with ``np.linalg.LinAlgError`` rather than truncate
    when they cannot certify a rank decision or represent a result.
    """
    try:
        yield
    except StructuredSolverError:
        raise
    except np.linalg.LinAlgError as error:
        raise StructuredSolverError(
            f"The structured solver cannot proceed: {str(error).rstrip('.')}. This should "
            "not happen for a model its data identify; direct_solve='gram' fits it with "
            "the dense solver instead."
        ) from error


# Levenberg shifts of an observed iterate the structured factor refuses as not
# positive definite (one-engine design §3.11): Wood, Pya & Saefken (2016, JASA,
# section 3.1.2) factor the Jacobi-preconditioned Hessian H' + eps I "with
# increasing eps, starting from zero, until positive definiteness is obtained";
# in unscaled coordinates that adds eps |H_ii| to each diagonal entry.  The
# sequence is fixed, so the shift an iterate takes is a function of the
# iterate alone; the perturbation does not move the converged mode.
_LEVENBERG_SHIFTS = tuple(10.0**power for power in range(-8, 9, 2))


def _levenberg_shifted_operator(
    penalized: NestedPenalizedOperator, shift: float
) -> tuple[NestedPenalizedOperator, NDArray]:
    """``penalized`` with ``E = shift diag(scale)`` added, and ``diag(E)`` ``(p,)``.

    The scale of a tree node is ``sum_{leaves under u} e_l + lambda_u`` (the
    leaves' weight-error mass, never below ``|H_uu|``, so a node whose signed
    weights cancel still moves) and of a border column ``|H_jj|``; the shift
    goes into the node penalties and the border penalty's diagonal, the parts
    of ``H`` the factor adds as they are.
    """
    data = penalized.data
    leaf = data.leaf
    mass = np.abs(leaf.weight) if leaf.error_mass is None else leaf.error_mass
    tree_scale = data.tree.subtree_sum(mass)
    node_shift = tuple(
        shift * (scale + lam) for lam, scale in zip(penalized.node_penalty, tree_scale, strict=True)
    )
    node_penalty = tuple(
        lam + step for lam, step in zip(penalized.node_penalty, node_shift, strict=True)
    )
    border_shift = shift * np.abs(np.diag(data.A) + np.diag(penalized.border_penalty))
    border_penalty = np.array(penalized.border_penalty, dtype=np.float64, copy=True)
    border_penalty[np.diag_indices_from(border_penalty)] += border_shift
    diagonal = np.empty(data.shape[0])
    diagonal[data.structured_indices] = np.concatenate(node_shift)
    diagonal[data.small_indices] = border_shift
    operator = NestedPenalizedOperator(
        data=data, node_penalty=node_penalty, border_penalty=border_penalty
    )
    return operator, diagonal


def _levenberg_shifted_leaf_operator(
    penalized: FactorSmoothPenalizedOperator | SumToZeroPenalizedOperator,
    shift: float,
    system: FactorSmoothLeafSystem | SumToZeroLeafSystem,
) -> tuple[
    FactorSmoothPenalizedOperator | SumToZeroPenalizedOperator,
    NDArray | Callable[[NDArray], NDArray],
]:
    """An ``fs`` (or ``sz``) operator with ``E = shift diag(scale)`` added, and ``diag(E)`` ``(p,)``.

    A level coordinate's scale is its rows' weight-error mass (the error
    Gram's diagonal of signed rows, else ``|D_jj|``) plus its penalty, never
    below ``|H_jj|``; a border column's is ``|H_jj|``.  The shift goes into
    the penalty parts the factor takes as square roots.  For ``sz`` the
    level shift is a per-level penalty of the level space (every one of the
    ``K`` levels), which the balance tree takes at its leaves; its public
    diagonal is the level's shift plus the last level's.
    """
    leaf = system.leaf
    k = leaf.block_size
    local = np.array(penalized.penalty_local, dtype=np.float64, copy=True)
    small = np.array(penalized.penalty_small, dtype=np.float64, copy=True)
    if leaf.error_diagonal is not None:
        mass = leaf.error_diagonal[:, :k]
    else:
        mass = np.abs(np.diagonal(system.operator.D, axis1=1, axis2=2))
    local_shift = shift * (mass + np.abs(np.diagonal(local, axis1=1, axis2=2)))
    border_shift = shift * np.abs(np.diag(penalized.A))
    index = np.arange(k)
    local[:, index, index] += local_shift
    small[np.diag_indices_from(small)] += border_shift
    operator = type(penalized)(
        A=system.operator.A + small,
        C=system.operator.C,
        D=system.operator.D + local,
        small_indices=penalized.small_indices,
        structured_indices=penalized.structured_indices,
        penalty_small=small,
        penalty_local=local,
    )
    structured = penalized.structured_indices
    if isinstance(penalized, SumToZeroPenalizedOperator):
        # E is diagonal in level space: E_pub beta = C' E_lev C beta, C = [I; -1']
        def apply(beta: NDArray) -> NDArray:
            levels = np.concatenate((beta[structured], -np.sum(beta[structured], axis=0)[None]))
            shifted = local_shift * levels
            out = np.empty_like(beta)
            out[structured] = shifted[:-1] - shifted[-1:]
            out[penalized.small_indices] = border_shift * beta[penalized.small_indices]
            return out

        return operator, apply
    diagonal = np.empty(penalized.shape[0])
    diagonal[structured] = local_shift
    diagonal[penalized.small_indices] = border_shift
    return operator, diagonal


def _build_iterate_factor(system, penalized_operator, *, observed: bool):
    """Factor one PIRLS iterate's structured system; return ``(factor, rhs, shift, diag(E))``.

    An ``sz`` shift is diagonal in level space, not in the public coordinates:
    its ``diag(E)`` is then a callable ``beta -> E beta``.

    Fisher rows are never refused for curvature (a structured refusal of them
    propagates).  An observed nested iterate the factor refuses (a tree pivot
    within its certified uncertainty, material negative border curvature)
    takes the smallest Levenberg shift of ``_LEVENBERG_SHIFTS`` whose factor
    certifies.  When none does, the unshifted refusal propagates, since it
    names the iterate's own cause; the largest shift's refusal is attached
    to it as a note.  The shifted Newton step ``(H + E)^-1 g`` from ``beta``
    is the normal-equations solve with ``E beta`` added to the right-hand
    side (the caller's part).
    """
    try:
        factor, rhs = build_augmented_structured_factor(system, penalized_operator)
        return factor, rhs, 0.0, None
    except np.linalg.LinAlgError as error:
        if not observed or not isinstance(
            penalized_operator,
            NestedPenalizedOperator | FactorSmoothPenalizedOperator | SumToZeroPenalizedOperator,
        ):
            raise
        refusal = error
    shifted_refusal: np.linalg.LinAlgError | None = None
    for shift in _LEVENBERG_SHIFTS:
        if isinstance(
            penalized_operator, FactorSmoothPenalizedOperator | SumToZeroPenalizedOperator
        ):
            shifted, diagonal = _levenberg_shifted_leaf_operator(penalized_operator, shift, system)
        else:
            shifted, diagonal = _levenberg_shifted_operator(penalized_operator, shift)
        try:
            factor, rhs = build_augmented_structured_factor(system, shifted)
            return factor, rhs, shift, diagonal
        except np.linalg.LinAlgError as error:
            shifted_refusal = error
    if shifted_refusal is not None:
        refusal.add_note(
            f"No Levenberg shift up to {_LEVENBERG_SHIFTS[-1]:g} certified the iterate either; "
            f"the largest shift's factor refused with: {shifted_refusal}"
        )
    raise refusal


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
    _retain_reml_decomposition: bool = False,
    _use_observed_newton: bool = True,
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
    _laplace_excluded: tuple[int, ...] = (),
    _mode_bar: float | None = None,
    _compensate_centred_intercept: bool = True,
    _centred_init: tuple[float, NDArray] | None = None,
) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
    """Fit by direct IRLS (see ``_fit_irls_direct_once``).

    A structured factor that cannot proceed raises ``StructuredSolverError``
    under every ``direct_solve`` (one-engine design §6, decision 6): no fit is
    rerun on another solver.  A Fisher-data reuse cache is cleared when the
    fit fails or ends unconverged.
    """
    result = None
    try:
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
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
            _scop_joint=_scop_joint,
            scop_state_init=scop_state_init,
            debug_recorder=debug_recorder,
            debug_context=debug_context,
            compute_rank_info=compute_rank_info,
            _return_working_system=_return_working_system,
            _compute_fit_statistics=_compute_fit_statistics,
            _compute_reml_geometry=_compute_reml_geometry,
            _retain_reml_decomposition=_retain_reml_decomposition,
            _use_observed_newton=_use_observed_newton,
            _deviance_init=_deviance_init,
            trace_run=trace_run,
            trace_purpose=trace_purpose,
            _compute_scop_postfit_inference=_compute_scop_postfit_inference,
            separation=separation,
            weight_semantics=weight_semantics,
            _initial_data_reuse=_initial_data_reuse,
            _raw_moment_policy=_raw_moment_policy,
            _fisher_data_reuse=_fisher_data_reuse,
            _laplace_excluded=_laplace_excluded,
            _mode_bar=_mode_bar,
            _compensate_centred_intercept=_compensate_centred_intercept,
            _centred_init=_centred_init,
        )
        return result
    finally:
        if _fisher_data_reuse is not None and (result is None or not result[0].converged):
            _fisher_data_reuse.clear()


def _fit_irls_direct_once(
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
    _retain_reml_decomposition: bool = False,
    _use_observed_newton: bool = True,
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
    _laplace_excluded: tuple[int, ...] = (),
    _mode_bar: float | None = None,
    _compensate_centred_intercept: bool = True,
    _centred_init: tuple[float, NDArray] | None = None,
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
        offset.  Ignored when ``intercept_init`` is not the start.
    _compute_reml_geometry : bool
        Internal SCOP-candidate switch. If False, omit the generic profiled
        slope inverse, determinant, and rank because the caller replaces them
        with one joint latent-coordinate LAML geometry. This requires both
        retained rank metadata and fit statistics to be disabled. Public and
        terminal fits must leave this True.
    _use_observed_newton : bool
        Internal curvature switch. When enabled, an ordinary Tweedie/log fit
        takes exact observed-Newton steps on every iteration, and a Gamma/log
        fit keeps Fisher scoring; no rejected step switches the curvature
        (one-engine design §3.11). Unsupported, constrained, SCOP, and
        cached-working-system routes retain Fisher scoring. Public callers
        should leave this True.
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
    objective_merit_scale = _irls_objective_scale(
        y=y,
        weights=weights,
        family=family,
        link=link,
    )

    structured_decision = resolve_structured_backend(
        gms,
        groups,
        direct_solve=direct_solve,
        coefficient_width=p,
        lambda2=lambda2,
        S_override=S_override,
        nesting_cache=getattr(dm, "_structured_layout_cache", None),
    )
    if structured_decision.use_structured and S_override is None and reml_penalties is None:
        reason = (
            "the structured candidate requires compact reml_penalties "
            "when S_override is not supplied"
        )
        if direct_solve == "structured":
            raise ValueError(f"direct_solve='structured' is ineligible: {reason}.")
        structured_decision = replace(
            structured_decision,
            use_structured=False,
            fallback_reason=reason,
        )
    _use_structured = structured_decision.use_structured
    _structured_group_index = structured_decision.group_index
    _direct_fallback_reason = structured_decision.fallback_reason
    _structured_layout = (
        get_structured_layout(
            dm,
            groups,
            dominant_group_index=_structured_group_index,
            chain_group_indices=structured_decision.chain_group_indices,
        )
        if _use_structured and _structured_group_index is not None
        else None
    )
    if isinstance(_structured_layout, FactorSmoothLeafLayout):
        # every dense kernel of the fs leaf route acts on its border (perf F7);
        # a nested chain's parent blocks are not border-wide and keep the release
        keep_narrow_cap(len(_structured_layout.small_indices) + 1)
    # One-engine design §3.8: a nested chain's PIRLS state is (alpha, beta)
    # about the fixed prior-weighted border centre c0 of its factor (0 on the
    # tree), so no eta is ever formed by cancelling X beta against a raw
    # intercept that absorbs a column's offset.
    _state_center: NDArray | None = None
    if isinstance(_structured_layout, NestedStructuredLayout):
        border_center, _ = nested_prior_statistics(_structured_layout, weights)
        _state_center = np.zeros(p)
        _state_center[_structured_layout.small_indices] = border_center
    elif isinstance(_structured_layout, FactorSmoothLeafLayout):
        # The same state for an fs term (design §3.4): (alpha, beta) about its c0.
        border_center, _ = factor_smooth_prior_statistics(_structured_layout, weights)
        _state_center = np.zeros(p)
        _state_center[_structured_layout.small_indices] = border_center

    if offset is None:
        offset = np.zeros(n)

    beta = beta_init.copy() if beta_init is not None else np.zeros(p)

    if intercept_init is not None:
        intercept = intercept_init
    else:
        intercept = coefficient_initial_intercept(
            distribution=family,
            link=link,
            y=y,
            sample_weight=weights,
        )

    # Dense paths retain the existing p x p penalty oracle. Structured paths
    # add each penalty directly to A or d, unless a caller already supplied a
    # dense override (which remains authoritative).
    S: NDArray | None
    if _use_structured:
        S = None if S_override is None else np.asarray(S_override, dtype=np.float64)
    elif S_override is not None:
        S = S_override
    else:
        S = _build_penalty_matrix(gms, groups, lambda2, p, reml_penalties=reml_penalties)

    def penalty_matvec(beta_values: NDArray, *, magnitude: bool = False) -> NDArray:
        """Apply the fitted penalty without expanding an identity random-effect block.

        With ``magnitude``, return ``|S| |beta|``, the magnitude this product
        sums, which bounds its rounding componentwise.
        """
        values = np.asarray(beta_values, dtype=np.float64)
        if magnitude:
            values = np.abs(values)
        if S is not None:
            return (np.abs(S) if magnitude else S) @ values
        if reml_penalties is None:  # pragma: no cover - validated above
            raise RuntimeError("Structured penalty components are unavailable.")
        from superglm.reml.penalty_algebra import (
            penalty_component_magnitude_matvec,
            penalty_component_matvec,
        )

        apply = penalty_component_magnitude_matvec if magnitude else penalty_component_matvec
        product = np.zeros_like(values)
        for component in reml_penalties:
            lam = float(lambda2[component.name]) if isinstance(lambda2, dict) else float(lambda2)
            if lam == 0.0:
                continue
            product[component.group_sl] += lam * apply(
                component,
                values[component.group_sl],
                gms[component.group_index],
            )
        return product

    def penalty_curvature() -> NDArray:
        """``diag(S)``: the dense penalty's diagonal, or the components' without forming ``S``."""
        if S is not None:
            return np.diag(S).astype(np.float64, copy=True)
        from superglm.reml.identified import penalty_diagonal

        lambdas = (
            lambda2
            if isinstance(lambda2, dict)
            else {component.name: float(lambda2) for component in reml_penalties or ()}
        )
        return penalty_diagonal(p, lambdas, reml_penalties)

    # diag(S), formed once per call on first use by the mode score (S is fixed)
    _penalty_curvature: list[NDArray] = []

    def penalty_quadratic(beta_values: NDArray) -> float:
        values = np.asarray(beta_values, dtype=np.float64)
        return float(values @ penalty_matvec(values))

    def mode_residual(
        beta_values: NDArray,
        intercept_value: float,
        mu_values: NDArray,
        eta_values: NDArray,
        centred_intercept: float | None = None,
    ) -> ModeResidual:
        """The observed-REML certificate's score at an iterate (``convergence="mode_score"``).

        The row score from Fisher rows (the score itself does not depend on
        the curvature the rows carry), centred on the system the last solve
        linearised about (``_score_centre``), its floors and weak tests
        resolved once every relative score is within ``MODE_RESOLVE_CAP``
        (``solvers.mode_score``).  The floors read the iterate's intercept
        about ``mean_x``; a centred state reads it from its own ``alpha`` and
        the offset of ``mean_x`` from the centre (``centre_offset_mean``), not
        from the raw intercept, which cancels ``c' beta`` at a column's offset.
        """
        assert _score_centre is not None
        mean_x, sum_w, diagonal, excluded_indices, offset_mean = _score_centre
        rows = coefficient_working_rows(
            distribution=family,
            link=link,
            y=y,
            mu=mu_values,
            eta=eta_values,
            sample_weight=weights,
            prefer_observed=False,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt(np.abs(diagonal) / sum_w)
        excluded = np.zeros(p, dtype=bool)
        excluded[list(excluded_indices)] = True
        # the slopes the REML fit's Laplace approximation leaves out
        # (``reml.identified``, design §3.9): flagged and kept, never gated
        excluded[list(_laplace_excluded)] = True
        if centred_intercept is None or offset_mean is None:
            shift = float(mean_x @ beta_values)
            alpha = float(intercept_value) + shift
            eta_tilde = eta_values - offset - float(intercept_value) - shift
        else:
            shift = float(offset_mean @ beta_values)
            alpha = float(centred_intercept) + shift
            eta_tilde = eta_values - offset - float(centred_intercept) - shift
        if not _penalty_curvature:
            _penalty_curvature.append(penalty_curvature())
        return penalized_mode_residual(
            dm=dm,
            row_score=rows.weights * (rows.response - eta_values),
            fisher_weights=rows.weights,
            positive_prior=weights > 0.0,
            mean_x=mean_x,
            centered_scale=np.where(np.isfinite(scale), scale, 0.0),
            alpha=alpha,
            eta_tilde=eta_tilde,
            penalty_score=penalty_matvec(beta_values),
            penalty_magnitude=penalty_matvec(beta_values, magnitude=True),
            penalty_curvature=_penalty_curvature[0],
            sum_w=sum_w,
            bar=mode_bar,
            excluded=excluded,
            resolve_cap=MODE_RESOLVE_CAP,
        )

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
        centred_intercept: float | None = None,
    ) -> _IRLSState:
        if trace_enabled:
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
                eta_unclipped = (
                    centred_intercept + centred_matvec(dm, beta_values, _state_center) + offset
                )
            intercept_value = centred_intercept - shift
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
        if centred_intercept is not None:
            state = replace(state, centred_intercept=float(centred_intercept))
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
        termination_reason: TerminationReason | None,
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
    # The feasibility scale needs ``|A_all|`` on every test, and the aggregate
    # system is fixed for the whole fit -- so it is built once here rather than
    # rebuilt two to three times per IRLS iteration plus once per line-search
    # halving.  See ``constrained_qp._feasibility_slack``.
    abs_A_all: NDArray | None = None
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
        abs_A_all = np.abs(A_all)

    # ── SCOP monotone engine support ──
    _has_scop = any(g.monotone_engine == "scop" for g in groups)
    if (
        _state_center is None
        and not _use_structured
        and not has_constraints
        and not _has_scop
        and hasattr(dm, "group_matrices")
    ):
        # The nested chain's centred state on gram's centred system too: eta
        # about the fixed prior-weighted centre, so a column at 1e8
        # contributes rows of its spread, not of its offset, and the
        # objective the line search compares keeps its digits (stage-1
        # verifier: a raw-1e8 column left the penalized deviance noisy at
        # 1e-11 relative, so no Newton step near the mode was accepted and
        # the score stalled at 1e-6).
        _state_center = prior_weighted_centre(dm, weights)
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
    # Laplace-approximate REML is built on the observed Hessian, so its PIRLS
    # runs on full-Newton weights (Wood 2011, JRSSB 73(1), section 3; Wood, Pya
    # & Saefken 2016, JASA, section 3.3): Fisher scoring shares the mode but
    # converges only linearly under a non-canonical link. Newton starts only
    # where the Fisher weights vary, so no constant-weight Gram or Fisher-data
    # cache exists for it to invalidate. Gamma/log keeps Fisher: its constant
    # Fisher weights reuse one weighted Gram that Newton would rebuild every
    # iteration; the mode certificate's bar is set by the REML tolerance
    # (``mode_score.mode_certification_bar``), which Fisher's linear rate
    # reaches.  The exported geometry is Fisher either way (``export_rows``),
    # and the weighted-Gram cache is off while Newton runs.  The curvature is
    # decided here, once, from the family, the link and the route (one-engine
    # design §3.11): no iterate switches it.
    # An observed iterate whose Hessian the structured factor refuses as not
    # positive definite takes a Levenberg shift inside the same curvature
    # (``_levenberg_shifted_operator``), and a trial whose observed rows are
    # not finite is rejected by the line search.
    _observed_newton_active = bool(
        _use_observed_newton
        and not has_constraints
        and not _has_scop
        and not _return_working_system
        and supports_observed_newton(family, link)
        and not _has_constant_irls_weights(family, link)
    )
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
                        "discarded_directions",
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
        initial_rows = coefficient_working_rows(
            distribution=family,
            link=link,
            y=y,
            mu=provisional.mu,
            eta=provisional.eta,
            sample_weight=weights,
            prefer_observed=False,
        )
        W_init = initial_rows.weights
        z_init = initial_rows.response
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
        # Constrained QP uses the shared stable centered-system builder, while
        # retaining the execution-plan route instead of materializing a
        # separate tabmat copy of the design.
        _tabmat_split = None
    else:
        # Ordinary intercept profiling currently benefits only when the split
        # contains a native high-cardinality categorical component. Avoid
        # materializing an unused dense duplicate for numeric and low-cardinality fits.
        _tabmat_split = dm.tabmat_centering_split
    _can_reuse_weighted_gram = _has_constant_irls_weights(family, link) and not _has_scop
    # Observed rows are not the constant Fisher weights this cache assumes.
    _can_reuse_weighted_gram = (
        _can_reuse_weighted_gram and not _use_structured and not _observed_newton_active
    )
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
                and (group.constraints is None or type(group.constraints) is LinearConstraintSet)
                for group in groups
            )
            and not (_use_qr or _use_structured or _has_scop)
            and debug_recorder is None
            and trace_run is None
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
    if profile is not None and _defer_raw_spline:
        profile["centered_spline_tabmat_cold_policy_rejections"] = (
            profile.get("centered_spline_tabmat_cold_policy_rejections", 0) + 1
        )
    _constant_centered_cache: CenteredSystem | None = None
    _constant_centered_z: NDArray | None = None
    _centered_factor_certification: _CenteredFactorCertification | None = None
    _initial_data_pending = (
        _initial_data_reuse is not None
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
        and debug_recorder is None
        and trace_run is None
    )
    # These objects stay live throughout the owner line search. A new design
    # or warm state cannot consume an entry from the previous coordinates.
    _initial_data_key = (
        (
            id(dm),
            id(dm.execution_plan),
            id(family),
            id(link),
            beta.tobytes(),
            np.float64(intercept).tobytes(),
        )
        if _initial_data_pending
        else None
    )

    def get_centered_system(W_current: NDArray, z_off_current: NDArray) -> CenteredSystem:
        nonlocal _constant_centered_cache, _constant_centered_z
        nonlocal _initial_data_pending
        reuse = _initial_data_reuse if _initial_data_pending else None
        _initial_data_pending = False
        before = replace(_tabmat_centering_state) if reuse is not None else None
        if reuse is not None:
            reused = reuse.take(
                _initial_data_key, W_current, z_off_current, _tabmat_centering_state, np.asarray(S)
            )
            if reused is not None:
                return reused
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
        fisher = (
            _fisher_data_reuse if _can_reuse_weighted_gram and not _observed_newton_active else None
        )
        data = (
            fisher.take((*fixed_owners, weight_semantics), W_current)
            if fisher is not None
            else None
        )
        system = build_centered_system(
            dm=dm,
            W=W_current,
            z_off=z_off_current,
            penalty=np.asarray(S),
            tabmat_split=_tabmat_split,
            tabmat_state=_tabmat_centering_state,
            profile=profile,
            _data=data,
        )
        if fisher is not None and fisher.data is None:
            fisher.remember(W_current, system)
        if _raw_moment_policy is not None and _tabmat_centering_state.raw_moment_eligible is False:
            _raw_moment_policy.raw_moment_eligible = False
        if reuse is not None:
            reuse.remember(
                _initial_data_key, W_current, z_off_current, before, _tabmat_centering_state, system
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
        if _fisher_data_reuse is not None:
            _fisher_data_reuse.clear()
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
    XtWX_beta: (
        NDArray | BlockSymmetricOperator | SumToZeroBlockOperator | NestedDataOperator | None
    ) = None
    _final_penalized_operator: (
        FactorSmoothPenalizedOperator | SumToZeroPenalizedOperator | NestedPenalizedOperator | None
    ) = None

    # Phase timing accumulators
    _t_working = 0.0
    _t_gram = 0.0
    _t_solve = 0.0
    _t_deviance = 0.0
    _t_eta = 0.0
    _t_deviance_eval = 0.0
    _last_working_centered: CenteredSystem | None = None
    # its mean_x less the state's centre (``centre_offset_mean``), None without one
    _last_working_offset_mean: NDArray | None = None
    _last_working_structured: (
        FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem | None
    ) = None
    # convergence="mode_score": the centring of the certificate's score at the
    # iterate each solve linearised about -- (mean_x, sum_w, centred diagonal,
    # coefficients the factor truncated as weakly identified, mean_x less the
    # state's centre or None without one) -- or None on a
    # route that forms no centred system (SCOP, linear constraints), which
    # keeps the coefficient-step test.
    _score_centre: tuple | None = None
    _last_mode_residual: ModeResidual | None = None
    # the certificate ratio at every iterate of a mode_score solve, for the
    # stagnation stop
    _mode_ratios: list[float] = []
    # the certificate's bar (``mode_score.mode_certification_bar``): the REML
    # fit passes the one its stopping tolerance needs
    mode_bar = MODE_CERTIFICATION_BAR if _mode_bar is None else float(_mode_bar)
    _stagnation_window = stagnation_window(max_iter, mode_bar)
    score_stagnated = False

    # The family's mean space, when the link's inverse can leave it (declared
    # by the family and link, ``irls_state.mean_space_violation``).
    _mean_space_invalid = mean_space_violation(family, link)
    # The dense columns whose centre lies beyond their spread (issue #430): only
    # with one does the raw intercept lose a bit, so the centred readings below
    # (a warm start's state, the gram intercept's mean offset, the certificate's
    # intercept, the sz score) run then, and a design without one computes as
    # before, bit for bit (``mode_score.offset_columns``).
    _offset_mask = None if _state_center is None else offset_columns(dm, weights, _state_center)
    _far_centre = _offset_mask is not None and bool(np.any(_offset_mask))
    # A warm start's centred state carried to this fit's centre (one-engine
    # design §3.8): alpha + (X - 1 c_warm') beta = alpha_c + (X - 1 c') beta
    # gives alpha_c = alpha + fsum((c - c_warm) beta), exact on every column
    # the two centres share.  Taken back from the raw intercept instead it
    # errs by u |c' beta| in every row's eta, ~0.25 at a 1e16 offset.
    centred_start: float | None = None
    if (
        _centred_init is not None
        and _far_centre
        and _state_center is not None
        and intercept_init is not None
        and intercept == intercept_init
    ):
        warm_alpha, warm_centre = _centred_init
        warm_centre = np.asarray(warm_centre, dtype=np.float64)
        if warm_centre.shape == _state_center.shape:
            centred_start = float(warm_alpha) + math.fsum((_state_center - warm_centre) * beta)
    # Freeze the fit-entry state so iteration-one trial safety has a baseline.
    committed = evaluate_state(
        beta,
        intercept,
        phase="initial",
        iteration=0,
        deviance=_deviance_init,
        emit_trace=not _has_scop,
        centred_intercept=centred_start,
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
                    discarded_directions=_immutable_or_none(st.get("discarded_directions")),
                )
                for gi, st in sorted(_scop_state.items())
            ),
        )

        # Remove mapped diagonal SCOP penalties before evaluating the merit.
        # Subtracting them after a full quadratic introduces avoidable
        # cancellation; cross-group override entries keep their existing role.
        scop_outer_penalty = S.copy()
        for spec in _scop_specs.values():
            scop_outer_penalty[spec.group.sl, spec.group.sl] = 0.0
        scop_outer_penalty_abs = np.abs(scop_outer_penalty)
        scop_merit_errors: dict[int, float] = {}
        merit_ku = (len(y) + 3 * len(beta) + 4 * len(_scop_specs) + 16) * (
            np.finfo(float).eps / 2.0
        )
        merit_gamma = merit_ku / (1.0 - merit_ku) if merit_ku < 1.0 else np.inf
        # For Gaussian deviance, (|y|+|mu|)^2 <= 8*y^2 + 2*(y-mu)^2.
        # This encloses residual formation without an observation-level
        # calculation on every trial. Other families supply deviance-unit
        # values; the guard covers their weighted accumulation and penalty.
        # Include gamma before squaring. The response action itself may be
        # outside binary64 even when its roundoff allowance is finite.
        with np.errstate(over="ignore", invalid="ignore"):
            gaussian_response_roundoff = (
                np.sum((np.sqrt(merit_gamma) * np.sqrt(weights) * y) ** 2)
                if type(family) is Gaussian
                else 0.0
            )

        def with_scop_merit(trial: _SCOPTrialState) -> _SCOPTrialState:
            """Attach deviance plus the latent-coordinate quadratic penalty."""
            penalty_quad = float(trial.irls.beta @ scop_outer_penalty @ trial.irls.beta)
            magnitude = np.abs(trial.irls.beta)
            penalty_roundoff = _positive_quadratic_roundoff(
                magnitude, scop_outer_penalty_abs, merit_gamma
            )
            for group_state in trial.groups:
                group = groups[group_state.group_index]
                lam_scop = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
                latent_penalty = _scop_specs[group_state.group_index].S_scop
                penalty_quad += float(
                    lam_scop * (group_state.beta_eff @ latent_penalty @ group_state.beta_eff)
                )
                magnitude = np.abs(group_state.beta_eff)
                penalty_roundoff += _positive_quadratic_roundoff(
                    magnitude, latent_penalty, merit_gamma, abs(lam_scop)
                )
            retained = replace(
                trial,
                irls=replace(
                    trial.irls,
                    penalized_deviance=float(trial.irls.deviance + penalty_quad),
                ),
            )
            deviance_roundoff = merit_gamma * abs(trial.irls.deviance)
            if type(family) is Gaussian:
                deviance_roundoff = 8.0 * gaussian_response_roundoff + 2.0 * deviance_roundoff
            with np.errstate(over="ignore", invalid="ignore"):
                allowance = float((deviance_roundoff + penalty_roundoff) / (1.0 - merit_gamma))
            scop_merit_errors[id(retained.irls)] = allowance
            return retained

        scop_committed = with_scop_merit(scop_committed)
        committed = scop_committed.irls
        # Relative change uses the current objective alone. An initial-merit
        # reference would weaken late convergence after a poor starting fit.
        objective_merit_scale = 0.0
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
    # Always an empty list; only ``record_diagnostics`` decides whether rows are
    # appended and whether it is published on the result.
    iteration_log: list[IterationDiagnostics] = []
    base_debug_context = dict(debug_context or {})
    # Level 2 fixes the row schema for the whole fit, so snapshot it at fit entry.
    record_debug_rows = (
        debug_recorder is not None and getattr(debug_recorder, "enabled_level", 0) >= 2
    )
    # Strictly wider than ``record_diagnostics``, so the per-iteration extrema
    # this gates are always bound by the time a diagnostics row is built.
    capture_extrema = record_diagnostics or record_debug_rows

    max_halving = 20  # max step-halving attempts per iteration
    _consecutive_svd = 0  # for auto-mode warning
    _reported_qp_nonconvergence = False  # transient note once; terminal authority is separate
    # A constrained fit-entry state has not been certified by the inner QP.
    retained_qp_converged = not has_constraints
    # Declared, not bound: the chain below is the whole set of reasons this
    # solver can end an iteration on, and declaring it against the shared
    # vocabulary is what ties the two together. A tenth reason invented here is
    # a type error on its own assignment until ``TerminationReason`` carries
    # it, so the consumers that switch on the field cannot be handed a string
    # they have never been shown.
    termination_reason: TerminationReason

    # Captured inside the loop, BEFORE ``dev_prev`` advances, because the
    # post-loop separation backstop needs the final iteration's movement.
    # Reading ``abs(dev - dev_prev)`` after the loop cannot work: ``dev_prev``
    # is assigned ``dev`` as the loop's last statement, so budget exhaustion
    # makes that difference identically zero and any stagnation test on it
    # vacuous.  ``inf`` until an iteration completes, so a fit that never
    # finishes one is never called stagnant.
    deviance_relative_change = float("inf")

    for it in range(max_iter):
        beta_prev = committed.beta
        intercept_prev = committed.intercept
        beta = committed.beta.copy()
        intercept = committed.intercept
        committed_active_set = None if prev_active_set is None else list(prev_active_set)
        proposal_qp_converged = not has_constraints
        # W/z are rebuilt below from the committed coefficients.  Any
        # constrained-QP certificate retained from the preceding iteration
        # therefore belongs to different working weights and response, not to
        # this iteration's H/g.  Only a certified full proposal from the
        # current working problem can restore the flag.
        retained_qp_converged = not has_constraints
        rank_truncated: bool | None = None
        used_rank_certification = False
        scop_proposal_eta_unclipped: NDArray | None = None
        proposal_centred_intercept: float | None = None

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
        if working_rows.rejection_reason is not None:
            # Every accepted trial of an observed fit had finite rows (the
            # line search rejects the others), so only the entry state can
            # get here: the declared curvature has no step to take.
            raise ValueError(
                "IRLS direct cannot start: the observed-Newton rows of "
                f"{type(family).__name__} with a {type(link).__name__} are not finite at "
                "the starting coefficients."
            )
        if working_rows.curvature_source == "observed" and profile is not None:
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
            _last_working_offset_mean = None
            if _state_center is not None:
                _last_working_offset_mean = centre_offset_mean(
                    dm, W, centered.sum_w, _state_center, centered.mean_x, _offset_mask
                )
                proposal_centred_intercept = centered.mean_z - math.fsum(
                    _last_working_offset_mean * beta
                )
                intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
            _cond_est = iteration_rank.pre_truncation_condition
            _used_svd = iteration_rank.used_svd_fallback
            rank_truncated = iteration_rank.rank_truncated
            if convergence == "mode_score":
                _score_centre = (
                    centered.mean_x,
                    centered.sum_w,
                    np.diag(centered.data_gram),
                    (),
                    _last_working_offset_mean if _far_centre else None,
                )
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
                reduced_rank = decompose_gram_if_authoritative(reduced_centered.hessian)
                reduced_factor_rhs = None
                if reduced_rank is None:
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
                structured_system = build_structured_system(
                    gms,
                    groups,
                    W,
                    Wz,
                    signed=working_rows.curvature_source == "observed",
                    dominant_group_index=_structured_group_index,
                    layout=_structured_layout,
                    prior_weights=weights,
                )
                penalized_operator = build_penalized_structured_operator(
                    structured_system,
                    gms,
                    groups,
                    lambda2,
                    reml_penalties=reml_penalties,
                    S_override=S_override,
                )
                _last_working_structured = structured_system
                _final_penalized_operator = penalized_operator
                _t_gram += time.perf_counter() - _t0

                _t0 = time.perf_counter()
                with _structured_solver_errors():
                    augmented_factor, rhs, levenberg_shift, shift_diagonal = _build_iterate_factor(
                        structured_system,
                        penalized_operator,
                        observed=working_rows.curvature_source == "observed",
                    )
                    beta_aug = solve_augmented_normal_equations(
                        structured_system,
                        augmented_factor,
                        rhs,
                        centred=_state_center is not None,
                        extra=(
                            None
                            if shift_diagonal is None
                            else np.concatenate(
                                (
                                    [0.0],
                                    shift_diagonal(committed.beta)
                                    if callable(shift_diagonal)
                                    else shift_diagonal * committed.beta,
                                )
                            )
                        ),
                    )
                if levenberg_shift and profile is not None:
                    profile["irls_levenberg_shifts"] = profile.get("irls_levenberg_shifts", 0) + 1
                    profile["irls_levenberg_max_shift"] = max(
                        float(profile.get("irls_levenberg_max_shift", 0.0)), levenberg_shift
                    )
                beta = beta_aug[1:]
                if _state_center is not None:
                    # the solve's intercept is the centred alpha about the same c0
                    proposal_centred_intercept = float(beta_aug[0])
                    intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
                else:
                    intercept = float(beta_aug[0])
                if augmented_factor.rank_truncated and isinstance(
                    augmented_factor, SumToZeroTreeFactor
                ):
                    # A truncated factor is a generalized inverse, and its
                    # normal-equations solution ``H^+ [1 X]'Wz`` is the minimum-norm
                    # representative: it resets the iterate's component along every
                    # truncated direction on each step.  Along an ``sz`` level whose
                    # rows the likelihood drives to the link's boundary (a one-row
                    # level with a response below a log link's range: no finite
                    # mode, Geyer 2009, Theorem 4) the direction is truncated once
                    # its rows' working weight falls to its rounding, and the reset
                    # moves those rows' eta arbitrarily far (+2844 from -20.7 on
                    # the stage-4 fixture); step halving then stalls every other
                    # coefficient and PIRLS ends uncertified.  The Gauss-Newton
                    # step of a rank-deficient problem is the minimum-norm
                    # *increment* (Pes & Rodriguez 2021, arXiv 2101.07560, eqs.
                    # 1.2-1.4; the Newton step of Wood, Pya & Safken 2016, section
                    # 3.1.2): ``Delta = H^+ g``, ``g = [1 X]' W (z - eta) - [0; S
                    # beta]``, the shifted factor's under a Levenberg shift, which
                    # leaves the iterate where it is along the truncated directions.
                    # Projecting the iterate onto the null space on every step (the
                    # minimal-norm variant, ibid. section 2, and in effect the
                    # solve above) is safe only along exact nulls; along a direction
                    # truncated as rank but carrying data it raises the residual,
                    # the failure they analyse.  Along an exact null (data and
                    # penalty both leave it free) the iterate keeps its component:
                    # no fitted value or penalty moves along it.  In exact
                    # arithmetic the same step as the solve above whenever nothing
                    # is truncated.
                    residual_rows = W * (z - eta)
                    gradient = np.empty(p + 1, dtype=np.float64)
                    gradient[0] = float(np.sum(residual_rows))
                    # the increment's intercept entry in the state's own coordinate:
                    # alpha about the centre when the state is centred (round 1's
                    # centred state; the raw intercept would cancel at a large
                    # column offset), else the raw intercept.  A centred state
                    # takes the score about the same centre, ``(X - 1 c')' r``
                    # formed on centred rows, so the border's ``X' r`` is never
                    # cancelled against ``c`` times the intercept's (a 1e16
                    # offset left no digit of it and the step was rejected)
                    if _far_centre and _state_center is not None:
                        gradient[1:] = centred_data_score(
                            dm, residual_rows, _state_center
                        ) - penalty_matvec(committed.beta)
                        increment = augmented_factor.solve(
                            gradient,
                            centred=True,
                            border_centred=gradient[augmented_factor.small_indices[1:]],
                        )
                    else:
                        gradient[1:] = dm.rmatvec(residual_rows) - penalty_matvec(committed.beta)
                        increment = augmented_factor.solve(
                            gradient, centred=_state_center is not None
                        )
                    beta = committed.beta + increment[1:]
                    if _state_center is not None:
                        assert committed.centred_intercept is not None
                        proposal_centred_intercept = float(committed.centred_intercept) + float(
                            increment[0]
                        )
                        intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
                    else:
                        intercept = float(committed.intercept) + float(increment[0])
                _used_svd = False  # one path: the engine has no SVD fallback
                _cond_est = augmented_factor.schur_condition_estimate
                rank_truncated = augmented_factor.rank_truncated
                if convergence == "mode_score":
                    _score_centre = _structured_score_centre(
                        structured_system, augmented_factor, dm, W, _state_center, _offset_mask
                    )
                _t_solve += time.perf_counter() - _t0
            elif not has_constraints:
                centered = get_centered_system(W, z_off)
                _last_working_centered = centered
                _t_gram += time.perf_counter() - _t0
                _t0 = time.perf_counter()
                iteration_rank = decompose_gram_if_authoritative(centered.hessian)
                iteration_factor_rhs = None
                if iteration_rank is None:
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
                _last_working_offset_mean = None
                if _state_center is not None:
                    # the intercept about the state's centre, from the offset
                    # of the working mean to it, formed on centred rows
                    # (``centre_offset_mean``): no raw-scale cancellation
                    _last_working_offset_mean = centre_offset_mean(
                        dm, W, centered.sum_w, _state_center, centered.mean_x, _offset_mask
                    )
                    proposal_centred_intercept = centered.mean_z - math.fsum(
                        _last_working_offset_mean * beta
                    )
                    intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
                _cond_est = iteration_rank.pre_truncation_condition
                _used_svd = iteration_rank.used_svd_fallback
                rank_truncated = iteration_rank.rank_truncated
                if convergence == "mode_score":
                    _score_centre = (
                        centered.mean_x,
                        centered.sum_w,
                        np.diag(centered.data_gram),
                        (),
                        _last_working_offset_mean if _far_centre else None,
                    )
                _t_solve += time.perf_counter() - _t0
            else:
                centered = get_centered_system(W, z_off)
                _last_working_centered = centered
                _t_gram += time.perf_counter() - _t0

                # Solve the constrained system in its existing coordinate space.
                _t0 = time.perf_counter()
                qp_result = _solve_constrained_qp_with_cold_retry(
                    centered.hessian,
                    centered.rhs,
                    A_all,
                    b_all,
                    prev_active_set,
                )
                beta = qp_result.beta
                intercept = centered.mean_z - float(centered.mean_x @ beta)
                proposal_qp_converged = bool(qp_result.converged)
                prev_active_set = qp_result.active_set
                # Non-convergence usually persists for the rest of the fit, so
                # latch the report to the first occurrence rather than emitting
                # one identical line per IRLS iteration. A later solve may
                # recover; the retained state's terminal authority is checked
                # independently below.
                if not qp_result.converged and not _reported_qp_nonconvergence:
                    _reported_qp_nonconvergence = True
                    logger.info(
                        "fit_irls_direct: constrained QP did not converge at "
                        "iteration %d; its KKT certificate is incomplete. "
                        "A later IRLS iteration may recover; subsequent inner "
                        "failures in this fit are not reported here.",
                        it + 1,
                    )
                # Report what the QP actually inverted.  These were hardcoded
                # to False/0.0, so the constrained branch published "perfectly
                # conditioned" to the consumers below while the unconstrained
                # branch above read the real numbers off its decomposition.
                _used_svd = qp_result.used_svd_fallback
                _cond_est = qp_result.condition
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
                        discarded_directions=_immutable_or_none(
                            scop_results[gi].discarded_directions
                        ),
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
                extended_max_halving=lambda: _poisson_sqrt_halving_budget(
                    committed=scop_committed.irls,
                    proposal=proposal_scop.irls,
                    y=y,
                    weights=weights,
                    family=family,
                    link=link,
                    default=max_halving,
                ),
                merit_scale=0.0,
                merit_roundoff=lambda candidate, current: (
                    scop_merit_errors[id(candidate)] + scop_merit_errors[id(current)]
                ),
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
                st["discarded_directions"] = (
                    None
                    if group_state.discarded_directions is None
                    else group_state.discarded_directions.copy()
                )
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
                centred_intercept=proposal_centred_intercept,
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
                centred_trial = None
                if (
                    committed.centred_intercept is not None
                    and proposal.centred_intercept is not None
                ):
                    centred_trial = committed.centred_intercept + alpha * (
                        proposal.centred_intercept - committed.centred_intercept
                    )
                candidate = evaluate_state(
                    beta_trial,
                    intercept_trial,
                    phase="line_search_trial",
                    iteration=it + 1,
                    alpha=alpha,
                    eta_unclipped=eta_trial,
                    enclosing_proposal_state_id=proposal.state_id,
                    centred_intercept=centred_trial,
                )
                trial_cache[alpha] = candidate
                return candidate

            committed_constraints_feasible = True
            proposal_constraints_feasible = True
            constraint_trial_is_invalid = None
            if _observed_newton_active:
                # one-engine design §3.11: finiteness is the one condition the
                # declared observed rows keep; a trial without finite rows is
                # not a state the next Newton step can start from

                def constraint_trial_is_invalid(candidate: _IRLSState) -> bool:
                    rows = coefficient_working_rows(
                        distribution=family,
                        link=link,
                        y=y,
                        mu=candidate.mu,
                        eta=candidate.eta,
                        sample_weight=weights,
                        prefer_observed=True,
                    )
                    return rows.rejection_reason is not None

            if has_constraints:
                if A_all is None or b_all is None:  # pragma: no cover - construction invariant
                    raise RuntimeError("constrained fit omitted its aggregate constraint system")
                committed_constraints_feasible = _is_feasible(
                    A_all,
                    committed.beta,
                    b_all,
                    _QP_FEASIBILITY_TOL,
                    abs_A=abs_A_all,
                )
                proposal_constraints_feasible = _is_feasible(
                    A_all,
                    proposal.beta,
                    b_all,
                    _QP_FEASIBILITY_TOL,
                    abs_A=abs_A_all,
                )

                def constraint_trial_is_invalid(candidate: _IRLSState) -> bool:
                    return not _is_feasible(
                        A_all,
                        candidate.beta,
                        b_all,
                        _QP_FEASIBILITY_TOL,
                        abs_A=abs_A_all,
                    )

            trial_is_invalid = constraint_trial_is_invalid
            if _mean_space_invalid is not None and not _mean_space_invalid(
                committed.eta_unclipped, weights
            ):
                # A feasible committed state keeps every accepted trial in the
                # family's mean space (``mean_space_violation``).
                def trial_is_invalid(
                    candidate: _IRLSState, other=constraint_trial_is_invalid
                ) -> bool:
                    assert _mean_space_invalid is not None
                    if _mean_space_invalid(candidate.eta_unclipped, weights):
                        return True
                    return other is not None and other(candidate)

            decision = _select_irls_trial(
                committed=committed,
                proposal=proposal,
                evaluate_state=evaluate_trial,
                invalid_state=trial_is_invalid,
                max_halving=max_halving,
                extended_max_halving=lambda: _poisson_sqrt_halving_budget(
                    committed=committed,
                    proposal=proposal,
                    y=y,
                    weights=weights,
                    family=family,
                    link=link,
                    default=max_halving,
                ),
                merit_scale=objective_merit_scale,
                merit_delta=lambda candidate, base: _stable_penalized_deviance_delta(
                    candidate,
                    base,
                    penalty_matvec,
                ),
            )
            if (
                has_constraints
                and decision.step_rejected
                and not committed_constraints_feasible
                and proposal_constraints_feasible
                and _state_is_finite(proposal)
            ):
                # The objective comparison is not allowed to prefer an
                # inadmissible state over a finite feasible QP proposal. Once
                # this proposal is committed, convexity of A beta >= b and the
                # invalid-state predicate above preserve feasibility on every
                # subsequent accepted line-search segment.
                decision = _IRLSStepDecision(
                    alpha=1.0,
                    step_halvings=0,
                    step_rejected=False,
                    trials_attempted=decision.trials_attempted,
                )
                logger.info(
                    "  irls_direct iter=%d: accepted finite feasible constrained "
                    "proposal after objective line search found no admissible trial",
                    it + 1,
                )
            retained = committed if decision.step_rejected else trial_cache[decision.alpha]
            if has_constraints:
                # Rejection retains coefficients from the preceding working
                # problem, while damping retains an interpolation.  Neither
                # state is the current H/g solution whose stationarity and
                # dual feasibility the inner QP certified.
                retained_qp_converged = bool(
                    not decision.step_rejected and decision.alpha == 1.0 and proposal_qp_converged
                )
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
            if convergence == "mode_score" and _score_centre is not None:
                # One-engine design §3.8: stop on the certificate's own score,
                # in centred coordinates over the identified coefficients,
                # once it meets the bar the certificate reads (no margin: the
                # certificate is this residual, ``mode_score``).  The step
                # length, with its raw intercept, is only reported.
                coef_change = float(
                    np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta)), initial=0.0)
                )
                residual = mode_residual(
                    beta, intercept, mu, eta, centred_intercept=retained.centred_intercept
                )
                convergence_value = mode_bar * residual.ratio()
                converged_this_iter = residual.ratio() <= 1.0
                _last_mode_residual = residual
                _mode_ratios.append(residual.ratio())
                # The score stopped contracting (``stagnation_window``):
                # the iterate is at its limiting accuracy, so the solve ends
                # uncertified and the caller discloses it.
                score_stagnated = (
                    not converged_this_iter
                    and residual.resolved
                    and len(_mode_ratios) > _stagnation_window
                    and min(_mode_ratios[-_stagnation_window:])
                    > 0.5 * min(_mode_ratios[:-_stagnation_window])
                )
            elif convergence in ("coefficients", "mode_score"):
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
                convergence_value = coef_change
                converged_this_iter = coef_change < tol
            else:
                objective = (
                    retained.deviance
                    if retained.penalized_deviance is None
                    else retained.penalized_deviance
                )
                if np.isfinite(objective_prev):
                    dev_rel_change = _irls_objective_relative_change(
                        objective=objective,
                        previous=objective_prev,
                        objective_scale=objective_merit_scale,
                    )
                converged_this_iter = dev_rel_change is not None and dev_rel_change < tol
                convergence_value = dev_rel_change
        else:
            converged_this_iter = False
            convergence_value = None
        if step_rejected:
            converged_this_iter = False

        constraints_feasible_this_iter = True
        if has_constraints:
            if A_all is None or b_all is None:  # pragma: no cover - construction invariant
                raise RuntimeError("constrained fit omitted its aggregate constraint system")
            constraints_feasible_this_iter = _is_feasible(
                A_all,
                beta,
                b_all,
                _QP_FEASIBILITY_TOL,
                abs_A=abs_A_all,
            )
            # Objective or coefficient stagnation cannot certify a constrained
            # mode whose retained coefficients are outside the feasible set.
            # Keep iterating while budget remains: a later QP/line-search step
            # may still repair an infeasible warm start.
            if not constraints_feasible_this_iter:
                converged_this_iter = False
            # Outer coefficient/deviance stagnation cannot replace the inner
            # QP's stationarity and dual-feasibility certificate. Keep a
            # finite feasible iterate and continue: a later full QP proposal
            # may still obtain the missing certificate.
            if not retained_qp_converged:
                converged_this_iter = False

        terminal_constraint_infeasible = bool(
            not constraints_feasible_this_iter
            and (step_rejected or not np.isfinite(dev) or it + 1 == max_iter)
        )
        terminal_constraint_kkt_incomplete = bool(
            has_constraints
            and not retained_qp_converged
            and (step_rejected or not np.isfinite(dev) or it + 1 == max_iter)
        )

        if terminal_constraint_infeasible:
            termination_reason = "constraint_infeasible"
        elif terminal_constraint_kkt_incomplete:
            termination_reason = "constraint_kkt_incomplete"
        elif step_rejected:
            termination_reason = "step_rejected"
        elif not np.isfinite(dev):
            termination_reason = "nonfinite_deviance"
        elif converged_this_iter:
            termination_reason = "converged"
        elif score_stagnated:
            termination_reason = "score_stagnated"
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
            # Each of these is read again by the diagnostics row below, which
            # only runs when ``capture_extrema`` is true (see its definition).
            # Bind once: they are O(n) passes over arrays nothing mutates
            # between here and the append.
            working_eta_min = float(np.min(working_eta))
            working_eta_max = float(np.max(working_eta))
            working_eta_min_unclipped = float(np.min(working_eta_unclipped))
            working_eta_max_unclipped = float(np.max(working_eta_unclipped))
            eta_min = float(np.min(eta))
            eta_max = float(np.max(eta))
            eta_min_unclipped = float(np.min(eta_unclipped))
            eta_max_unclipped = float(np.max(eta_unclipped))
            working_eta_clipped = bool(
                working_eta_min_unclipped < working_eta_min
                or working_eta_max_unclipped > working_eta_max
            )
            eta_clipped = bool(eta_min_unclipped < eta_min or eta_max_unclipped > eta_max)

        # Record per-iteration diagnostics
        if record_diagnostics:
            top_idx, bot_idx = _extreme_weight_indices(W)
            w_min = float(W.min())
            w_max = float(W.max())
            iteration_log.append(
                IterationDiagnostics(
                    iteration=it + 1,
                    deviance=dev,
                    w_min=w_min,
                    w_max=w_max,
                    w_ratio=w_ratio,
                    mu_min=float(mu.min()),
                    mu_max=float(mu.max()),
                    eta_min=eta_min,
                    eta_max=eta_max,
                    intercept=intercept,
                    step_halvings=n_halvings,
                    top_w_indices=top_idx,
                    bottom_w_indices=bot_idx,
                    cond_estimate=_cond_est,
                    used_svd_fallback=_used_svd,
                    raw_w_min=w_min,
                    raw_w_max=w_max,
                    raw_w_ratio=w_ratio,
                    eta_min_unclipped=eta_min_unclipped,
                    eta_max_unclipped=eta_max_unclipped,
                    eta_clipped=eta_clipped,
                    working_mu_min=float(working_mu.min()),
                    working_mu_max=float(working_mu.max()),
                    working_eta_min=working_eta_min,
                    working_eta_max=working_eta_max,
                    working_eta_min_unclipped=working_eta_min_unclipped,
                    working_eta_max_unclipped=working_eta_max_unclipped,
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

        deviance_relative_change = abs(dev - dev_prev) / (abs(dev_prev) + 1)
        logger.info(
            f"  irls_direct iter={it + 1:3d}  "
            f"dev={dev:12.1f}  delta={deviance_relative_change:10.2e}"
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
                },
            )

        if step_rejected:
            logger.warning(
                "IRLS direct rejected all trial steps at iter=%d; restored committed state",
                it + 1,
            )
            break

        committed = retained
        if _has_scop:
            scop_committed = retained_scop
        if converged_this_iter:
            converged = True
            break
        if score_stagnated:
            break
        dev_prev = dev
        objective_prev = (
            retained.deviance
            if retained.penalized_deviance is None
            else retained.penalized_deviance
        )

    t_elapsed = time.perf_counter() - t_start
    logger.info(f"  IRLS direct done: {it + 1} iters, {t_elapsed:.2f}s")

    # A published Gaussian identity fit carries its centred intercept as the
    # compensated pair (alpha, alpha_lo): alpha's weighted mean rounds by a
    # kernel-dependent ulp or more, and alpha + (x - c) beta ties at every row
    # when alpha* is the midpoint of two adjacent floats, so a one-ulp signal
    # between two levels collapsed on some BLAS kernels.  The deviance and
    # scale published below are the compensated predictor's, the eta every
    # consumer reads (``mode_score.linear_predictor``).
    centred_intercept_lo: float | None = None
    if (
        _compensate_centred_intercept
        and _compute_fit_statistics
        and retained.centred_intercept is not None
        and _state_center is not None
        and type(family) is Gaussian
        and type(link) is IdentityLink
    ):
        contribution = centred_matvec(dm, beta, _state_center)
        centred_intercept_lo = centred_intercept_remainder(
            y, weights, offset, retained.centred_intercept, contribution
        )
        if centred_intercept_lo is not None:
            contribution += centred_intercept_lo
            eta_unclipped = (retained.centred_intercept + contribution) + offset
            eta = stabilize_eta(eta_unclipped, link)
            mu = clip_mu(link.inverse(eta), family)
            dev = float(np.sum(weights * family.deviance_unit(y, mu)))

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
    if separation != "ignore" and np.isfinite(dev):
        from superglm.diagnostics.separation import (
            EXTREME_WEIGHT_RATIO,
            STAGNANT_DEVIANCE_DELTA,
            SeparationError,
            SeparationWarning,
            format_runtime_message,
        )

        separation_ratio = _separation_weight_ratio(
            working_rows.curvature_source,
            w_ratio,
            family=family,
            link=link,
            mu=working_mu,
            eta=working_eta,
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
                and max_iter >= 10
                and it + 1 >= max_iter
                and not step_rejected
                and deviance_relative_change < STAGNANT_DEVIANCE_DELTA
            )
            if pinned or exhausted_stagnant:
                max_abs = float(np.max(np.abs(beta))) if beta.size else 0.0
                drifting = [
                    g.name
                    for g in groups
                    if beta[g.sl].size and float(np.max(np.abs(beta[g.sl]))) >= max_abs - 2.0
                ][:5]
                message = format_runtime_message(separation_ratio, it + 1, drifting, pinned)
                if separation == "error":
                    raise SeparationError(message)
                warnings.warn(message, SeparationWarning, stacklevel=2)

    if has_constraints:
        if A_all is None or b_all is None:  # pragma: no cover - construction invariant
            raise RuntimeError("constrained fit omitted its aggregate constraint system")
        if not _is_feasible(A_all, beta, b_all, _QP_FEASIBILITY_TOL, abs_A=abs_A_all):
            minimum_scaled_slack = float(np.min(_feasibility_slack(A_all, beta, b_all)))
            converged = False
            termination_reason = "constraint_infeasible"
            logger.warning(
                "fit_irls_direct: retained coefficient mode violates hard constraints "
                "(minimum scaled slack %.3e, tolerance %.1e); fit is not converged.",
                minimum_scaled_slack,
                _QP_FEASIBILITY_TOL,
            )
        elif not retained_qp_converged:
            converged = False
            termination_reason = "constraint_kkt_incomplete"
            logger.warning(
                "fit_irls_direct: retained coefficient mode has no complete "
                "constrained-QP KKT certificate; fit is not converged."
            )

    # A returned state with rows at the mean-space boundary is not a mode of
    # the model: their capped mean makes the deviance flat there, and the
    # maximum it approaches is constrained, not stationary.  It is never
    # converged, whatever stopped the loop (``mean_space_boundary_rows``).
    _boundary_rows = (
        0
        if _mean_space_invalid is None
        else mean_space_boundary_rows(family, link, eta_unclipped, weights)
    )
    if _boundary_rows:
        converged = False
        termination_reason = "mean_space_boundary"
        logger.info(
            "fit_irls_direct: %d row(s) at the boundary of the family's mean space; "
            "the penalized maximum is constrained, fit is not converged.",
            _boundary_rows,
        )

    if _has_scop:
        # Final Gram and SCOP Hessian caches must describe the retained model,
        # not the working state or a discarded full proposal.
        final_rows = coefficient_working_rows(
            distribution=family,
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=weights,
            prefer_observed=False,
        )
        W = final_rows.weights
        z = final_rows.response

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
                    _scop_state[gi]["discarded_directions"] = (
                        None
                        if refresh.discarded_directions is None
                        else refresh.discarded_directions.copy()
                    )
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
                    st["discarded_directions"] = (
                        None
                        if refresh.discarded_directions is None
                        else refresh.discarded_directions.copy()
                    )

    # Every public/exported matrix and rank claim is evaluated at the retained
    # model. Private fREML performance iterations deliberately reuse the
    # working system used for their one coefficient update.
    if not _return_working_system:
        export_rows = coefficient_working_rows(
            distribution=family,
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=weights,
            prefer_observed=False,
        )
        W = export_rows.weights
        z = export_rows.response

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
        if _boundary_rows:
            profile["irls_mean_space_boundary_calls"] = (
                profile.get("irls_mean_space_boundary_calls", 0) + 1
            )
        if _last_mode_residual is not None:
            # §3.8, §3.9: the stop rule's last score, whether a derived floor
            # bound it, and the slopes it flagged weakly identified and kept.
            profile["irls_mode_score"] = mode_bar * _last_mode_residual.ratio()
            profile["irls_mode_bar"] = mode_bar
            profile["irls_mode_floor_binding"] = _last_mode_residual.floor_binding
            profile["irls_mode_weakly_identified"] = tuple(
                int(index) for index in np.flatnonzero(_last_mode_residual.excluded)
            )

    # Preserve the raw coefficient-space payload used by REML separately from
    # the centered inference system. The structured path retains the same
    # moments in block form and never materializes the dominant K x K block.
    centered_final: CenteredSystem | None = None
    structured_final: (
        FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem | None
    ) = None
    if _use_structured:
        if _return_working_system:
            if _last_working_structured is None:
                raise RuntimeError("working structured system was not computed")
            structured_final = _last_working_structured
        else:
            if _structured_group_index is None:  # pragma: no cover - selection invariant
                raise RuntimeError("Structured backend has no dominant group.")
            z_off = z - offset
            structured_final = build_structured_system(
                gms,
                groups,
                W,
                W * z_off,
                signed=export_rows.curvature_source == "observed",
                dominant_group_index=_structured_group_index,
                layout=_structured_layout,
                prior_weights=weights,
            )
        _final_penalized_operator = build_penalized_structured_operator(
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
        offset_mean_final = None
    else:
        if _return_working_system:
            if _last_working_centered is None:
                raise RuntimeError("working centered system was not computed")
            centered_final = _last_working_centered
            offset_mean_final = _last_working_offset_mean
        else:
            z_off = z - offset
            centered_final = get_centered_system(W, z_off)
            offset_mean_final = (
                None
                if _state_center is None or cache_out is None
                else centre_offset_mean(
                    dm, W, centered_final.sum_w, _state_center, centered_final.mean_x, _offset_mask
                )
            )
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
            if not _return_working_system and isinstance(structured_final, FactorSmoothLeafSystem):
                # The retained estimability rule reads these (a nested chain
                # certifies its border on its own leaf-shifted moments).
                cache_out["structured_row_column_norm"] = cancelled_column_row_norms(
                    centred_data_operator(structured_final), dm, W
                )
        else:
            if centered_final is None or XtWX is None:  # pragma: no cover - branch invariant
                raise RuntimeError("Dense fit did not produce a centered system.")
            cache_out["XtWX"] = XtWX
            cache_out["centered_XtWX"] = centered_final.data_gram
            # mean_x less the state's centre, formed on centred rows: a cached
            # trial's centred intercept (``reml.discrete``) reads it
            cache_out["centre_offset_mean"] = offset_mean_final
            # the matrix the slope decomposition was taken of (the identified
            # part of the Laplace approximation restricts it, ``reml.identified``)
            cache_out["centered_hessian"] = centered_final.hessian
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
    structured_factor: (
        ProfiledFactorSmoothLeafFactor
        | ProfiledSumToZeroTreeFactor
        | ProfiledNestedSchurFactor
        | None
    ) = None
    reml_geometry_summary: REMLGeometrySummary | None = None
    if _use_structured:
        if structured_final is None or _final_penalized_operator is None:
            raise RuntimeError("Structured fit did not produce final coefficient blocks.")
        with _structured_solver_errors():
            augmented_factor, _ = build_augmented_structured_factor(
                structured_final, _final_penalized_operator
            )
        if isinstance(augmented_factor, SumToZeroTreeFactor):
            structured_factor = ProfiledSumToZeroTreeFactor(
                augmented_factor=augmented_factor,
                sum_w=structured_final.sum_w,
                xtw=XtW1,
            )
        elif isinstance(augmented_factor, FactorSmoothLeafFactor):
            structured_factor = ProfiledFactorSmoothLeafFactor(
                augmented_factor=augmented_factor,
                sum_w=structured_final.sum_w,
                xtw=XtW1,
            )
        elif isinstance(augmented_factor, NestedSchurFactor) and isinstance(
            structured_final, NestedStructuredSystem
        ):
            structured_factor = ProfiledNestedSchurFactor(
                augmented_factor=augmented_factor,
                sum_w=structured_final.sum_w,
                xtw=XtW1,
                data_operator=structured_final.operator,
            )
        else:  # pragma: no cover - structured dispatch invariant
            raise TypeError("Unsupported structured factor geometry.")
        XtWX_beta = structured_final.operator
        if _compute_reml_geometry:
            XtWX_S_inv_beta: NDArray | HessianFactor = structured_factor
            with _structured_solver_errors():
                log_det_H: float | None = augmented_factor.logdet()
            reml_hessian_rank: int | None = augmented_factor.rank
        else:
            XtWX_S_inv_beta = np.empty((0, 0), dtype=np.float64)
            log_det_H = None
            reml_hessian_rank = None
        # Structured Schur factors have their own retained-factor protocol;
        # the dense-decomposition seam does not apply here.
        retained_reml_decomposition: RankDecomposition | None = None
        if _compute_fit_statistics:
            edf_operator = (
                CenteredBlockOperator(
                    raw=XtWX_beta,
                    cross=XtW1,
                    total=sum_W,
                    center=mean_x,
                )
                if isinstance(
                    XtWX_beta,
                    BlockSymmetricOperator | SumToZeroBlockOperator | NestedDataOperator,
                )
                else XtWX_beta
            )
            with _structured_solver_errors():
                p_eff = 1.0 + structured_factor.trace_inverse_operator(edf_operator)
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
            reml_slope_rank = decompose_gram_if_authoritative(centered_final.hessian)
            if reml_slope_rank is None:
                certification = certify_centered_factor(
                    centered_final,
                    W,
                )
                reml_slope_rank = certification.decomposition
            XtWX_S_inv_beta = reml_slope_rank.pseudo_inverse()
            log_det_H = float(np.log(centered_final.sum_w) + reml_slope_rank.log_pdet)
            reml_hessian_rank = 1 + reml_slope_rank.rank
            # O(p) summary of the centered system for the REML gradient and
            # objective, so the loop can run without per-fit rank metadata.
            # column_scale matches decompose_gram's definition bit-for-bit.
            reml_geometry_summary = REMLGeometrySummary(
                mean_x=np.asarray(centered_final.mean_x, dtype=np.float64),
                sum_w=float(centered_final.sum_w),
                column_scale=np.sqrt(np.maximum(np.diag(centered_final.data_gram), 0.0)),
            )
        else:
            reml_slope_rank = None
            XtWX_S_inv_beta = np.empty((0, 0), dtype=np.float64)
            log_det_H = None
            reml_hessian_rank = None
        retained_reml_decomposition = reml_slope_rank if _retain_reml_decomposition else None

        coefficient_rank = None
        if _compute_fit_statistics and compute_rank_info:
            M_beta = centered_final.hessian + centered_final.sum_w * np.outer(
                centered_final.mean_x, centered_final.mean_x
            )
            coefficient_rank = decompose_gram_if_authoritative(M_beta)
            if coefficient_rank is None:
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
                # The export rows are Fisher (``prefer_observed=False``), whose
                # weights are nonnegative: the data Gram is PSD by construction,
                # so a negative eigenvalue at its null is formation rounding and
                # sends the rank to the factor below instead of raising.
                data_rank = (
                    decompose_gram_if_authoritative(
                        centered_final.data_gram,
                        psd_by_construction=not _return_working_system,
                    )
                    if compute_rank_info
                    else None
                )
                if compute_rank_info and data_rank is None:
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
        # Repeated inner solves stay quiet; the REML drivers own the INFO line.
        record_auto_backend_decision(profile, direct_solve, structured_decision, log=False)
        if structured_factor is not None:
            profile["structured_dominant_group"] = structured_factor.dominant_group_name
            profile["structured_chain"] = tuple(
                groups[index].name for index in structured_decision.chain_group_indices
            )
            profile["structured_nested_fallback_reason"] = (
                structured_decision.nested_fallback_reason
            )
            border_certificate = getattr(structured_factor, "border_certificate", None)
            if border_certificate is not None:
                # §3.6 step 5: every null direction the border factorization
                # truncated, disclosed with its certificate.
                profile["structured_border_rank"] = border_certificate.rank
                profile["structured_border_tau"] = border_certificate.tau
                profile["structured_border_verification_decrements"] = border_certificate.decrements
                profile["structured_border_trailing_bound"] = border_certificate.trailing_bound
                profile["structured_border_deflated"] = border_certificate.deflated
                profile["structured_weakly_identified"] = tuple(
                    getattr(structured_factor, "weakly_identified_slopes", ())
                )
            profile["structured_minimum_local_diagonal"] = structured_factor.minimum_local_diagonal
            profile["structured_schur_condition"] = structured_factor.schur_condition_estimate

    # Pearson-based phi for estimated-scale families.  The numerator is the
    # same under either weight contract; only the denominator's likelihood
    # size distinguishes them.
    if _compute_fit_statistics and not getattr(family, "scale_known", True):
        pearson_sum = pearson_chi2(distribution=family, y=y, mu=mu, sample_weight=weights)
        df_resid = pearson_residual_degrees_of_freedom(
            weights,
            p_eff,
            weight_semantics=weight_semantics,
        )
        phi = pearson_sum / df_resid
    else:
        phi = 1.0
    if not _compute_reml_geometry or not _compute_fit_statistics:
        # Private SCOP candidates and stats-skipping in-loop REML fits have no
        # retained-fit statistics. NaN makes accidental publication fail
        # visibly instead of presenting 0 EDF or unit dispersion as if either
        # had been evaluated; published statistics come from the terminal
        # refit, which always computes them.
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
        reml_geometry=reml_geometry_summary,
        reml_slope_decomposition=retained_reml_decomposition,
        rank_info=rank_info,
        state_id=retained.state_id,
        evaluation_id=retained.evaluation_id,
        state_space=retained.state_space,
        basis_id=retained.basis_id,
        termination_reason=termination_reason,
        direct_backend=_resolved_direct_backend,
        direct_fallback_reason=_direct_fallback_reason,
        centred_intercept=retained.centred_intercept,
        state_center=None if retained.centred_intercept is None else _state_center,
        centred_intercept_lo=centred_intercept_lo,
        mean_space_boundary_rows=_boundary_rows,
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
                "discarded_directions": st.get("discarded_directions"),
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
            pearson_sum = pearson_chi2(distribution=family, y=y, mu=mu, sample_weight=weights)
            result.phi = pearson_sum / pearson_residual_degrees_of_freedom(
                weights,
                inference.total_edf,
                weight_semantics=weight_semantics,
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
