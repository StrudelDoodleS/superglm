"""SCOP-aware EFS REML optimizer.

Extends the standard Fellner-Schall EFS loop to handle SCOP monotone terms
alongside unconstrained SSP terms.

References
----------
Wood & Fasiolo (2017). A generalized Fellner-Schall method for smoothing
parameter optimization with application to shape constrained regression.
Biometrics 73(4), 1071-1081.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import stabilize_eta
from superglm.reml.objective import reml_laml_objective
from superglm.reml.observed_geometry import (
    ObservedGeometryInfeasibleError,
    ObservedModeNotCertifiedError,
    ObservedModeNotConvergedError,
    classify_scop_reml_curvature,
)
from superglm.reml.penalty_algebra import (
    _attach_context_geometry,
    _context_geometry,
    _frozen_array,
    build_penalty_matrix,
    compute_logdet_s_derivatives,
    penalty_component_dense_matrix,
    penalty_component_matvec,
    penalty_component_quadratic,
    penalty_component_trace,
)
from superglm.reml.result import REMLResult
from superglm.reml.scale import (
    prepare_reml_scale_data,
)
from superglm.reml.scop_geometry import (
    SCOPJointGeometry,
    SCOPModeScore,
    build_cached_scop_joint_geometry,
    build_observed_scop_joint_geometry,
    install_scop_postfit_inference,
    restrict_to_scop_resolved_range,
    scop_penalized_mode_score,
    scop_resolved_range_projector,
)
from superglm.solvers.centered_system import (
    grouped_augmented_factor,
    grouped_weighted_factor,
)
from superglm.solvers.dispersion import dispersion_likelihood_size
from superglm.solvers.irls_direct import SCOPRunCentring, fit_irls_direct
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankInfo,
    decompose_factor,
    decompose_gram_if_authoritative,
)
from superglm.solvers.working_rows import fisher_working_weights
from superglm.types import GroupSlice, PenaltyComponent

# These private thresholds intentionally mix units: absolute lambda scale for the
# floor guard, log-lambda-step scale for stability/plateau checks, and relative
# objective scale for outer-loop flatness checks.
_MULTI_SCOP_DISCRETE_LAMBDA_FLOOR = 1.0e-4
_MULTI_SCOP_DISCRETE_FLOOR_FACTOR = 1.05
_MULTI_SCOP_DISCRETE_LOG_STEP_TOL = 1.0e-3
_MULTI_SCOP_DISCRETE_MIN_STABLE_ITERS = 3
_MULTI_SCOP_DISCRETE_ACTIVE_PLATEAU_TOL = 5.0e-3
_MULTI_SCOP_DISCRETE_OBJ_REL_TOL = 1.0e-6
# The plateau exit is the step engine's achievable-precision endgame: it may
# grant convergence only once EFS steps have STOPPED contracting, i.e.
# further iterations no longer buy precision. While steps still contract,
# the loop continues toward the strict ``max_change < reml_tol`` road
# instead of being pre-empted at the plateau's fixed thresholds. Calibration
# (2026-08-07, monotone PSpline + cr smooth): at the machinery noise floor
# steps go 1.9e-5 -> 2.0e-5 (ratio 1.05, a genuine stall set by inner-mode
# noise), while a steadily linear EFS tail contracts at ratio ~0.6 with the
# lambda still walking one percent per iteration -- so the stall bar sits
# at 0.9, and two consecutive stalled observations are required so one
# noisy non-contraction cannot grant convergence mid-progress.
#
# Stall evidence is a bounded band, not a per-step ratio. A consecutive
# ratio counter got two real trajectories wrong: an oscillating noise
# floor (1e-5, 2e-5, 1e-5, ...) resets on every down-leg and never
# plateaus, exhausting max_reml_iter on a flat fit, while an expanding
# tail (0.002, 0.004, 0.008 -- the same-sign adaptive alpha deliberately
# grows) counts every ratio above the bar as a stall and plateaus while
# movement accelerates. Stalled therefore means: the last
# (MIN_STALLED_ITERS + 1) accepted steps all sit within STALL_BAND of
# their own minimum (the measured noise stall oscillates at ratio ~1.05,
# well inside 2x; a genuinely expanding tail leaves the band).
#
# A banded ratio just under 1 is still a geometric tail: at r=0.95 with
# max_change at the 0.01 plateau cap, max_change*r/(1-r) says ~19% of the
# log-lambda movement remains, and a "stalled" verdict there would forfeit
# it. The remaining-movement bound therefore defers the plateau in the
# r in [0.9, 1) band until the extrapolated tail is genuinely small. At
# r >= 1 no geometric extrapolation exists, and the band has already
# certified the steps as noise-bounded.
_SCOP_EFS_PLATEAU_MIN_STALLED_ITERS = 2
_SCOP_EFS_PLATEAU_STALL_BAND = 2.0
_SCOP_EFS_PLATEAU_REMAINING_MOVEMENT = 0.05
# Expansion means the LAST step sits materially above the window's
# minimum -- not any strict increase (equal steps carry 1e-16-relative
# exp/log jitter that can order itself increasingly and must stall) and
# not only monotone growth (a sawtooth 0.004, 0.0039, 0.006 hides a +54%
# step behind one down-tick). The threshold sits just above the largest
# LEGITIMATE stall leg the machinery measures: the 400-row EFS noise
# floor oscillates within ~5% per step, but the multi-SCOP cleanup
# endgame stalls in a recurring limit cycle whose growth legs run at
# ratio 1.199 per step (measured window [1.05e-4, 1.26e-4, 1.51e-4] with
# a flat objective) -- tightening the bar below that defers a real stall
# to max_reml_iter. Sustained growth under the 1.25 single-window
# resolution is therefore indistinguishable from the machinery's own
# stall vocabulary BY MEASUREMENT; in that band the plateau's
# remaining-movement cap is the guarantee, not the detector.
_SCOP_EFS_PLATEAU_EXPANSION_GROWTH = 1.25


logger = logging.getLogger(__name__)


def _scop_plateau_remaining_movement_bounded(max_change: float, contraction_ratio: float) -> bool:
    """Geometric-tail bound for the plateau's stalled verdict."""
    if contraction_ratio >= 1.0:
        return True
    remaining = max_change * contraction_ratio / (1.0 - contraction_ratio)
    return remaining < _SCOP_EFS_PLATEAU_REMAINING_MOVEMENT


def _scop_plateau_steps_stalled(accepted_changes: list[float], contraction_ratio: float) -> bool:
    """Stalled: a banded, non-expanding window with bounded remaining movement.

    Material recent growth is expansion however tightly banded, and it
    does not need every transition to increase: 0.004, 0.0039, 0.006 hides
    a +54% step behind one transient down-tick. The rule is therefore the
    LAST step against the window's minimum -- monotone growth, sawtooth
    growth, and the gradual in-band expander all defer until the
    trajectory turns, while 1e-16 exp/log jitter on equal steps and a
    genuine oscillation (last near its floor) stall. Deferral costs an
    iteration; a wrong plateau forfeits real movement.
    """
    if len(accepted_changes) < _SCOP_EFS_PLATEAU_MIN_STALLED_ITERS + 1:
        return False
    window = accepted_changes[-(_SCOP_EFS_PLATEAU_MIN_STALLED_ITERS + 1) :]
    floor = max(min(window), np.finfo(float).tiny)
    if max(window) > _SCOP_EFS_PLATEAU_STALL_BAND * floor:
        return False
    if window[-1] >= _SCOP_EFS_PLATEAU_EXPANSION_GROWTH * floor:
        return False
    return _scop_plateau_remaining_movement_bounded(window[-1], contraction_ratio)


# Safeguarded Aitken extrapolation of the EFS linear phase (perf_scout_scop
# prototype, measured in fix_reml_fit_time's variant study). Near its fixed point
# Fellner-Schall converges linearly (Wood & Fasiolo 2017, corollary of Theorem 3):
# a component whose accepted steps shrink by a stable ratio r has step / (1 - r)
# of movement left, which Aitken's delta-squared takes in one step. It fires only
# with four same-sign accepted steps whose three ratios lie in (0.3, 0.98) and
# agree within 0.05, a last step under 0.25, and a jump capped at 1.0 in log
# lambda; the history restarts after a jump. The cap was set when the decrease
# hold read tr(H^-1 S_j) alone, which made a jump past it absorbing: an unguarded
# jump ended a measured fit 5% off. The line search still accepts or rejects the step, and the
# stopping rules are unchanged.
_SCOP_EFS_AITKEN_MIN_RATIO = 0.3
_SCOP_EFS_AITKEN_MAX_RATIO = 0.98
_SCOP_EFS_AITKEN_RATIO_AGREEMENT = 0.05
_SCOP_EFS_AITKEN_MAX_LAST_STEP = 0.25
_SCOP_EFS_AITKEN_MAX_JUMP = 1.0

_SCOP_EFS_MAX_BACKTRACK_ATTEMPTS = 8
_SCOP_EFS_MAX_REFLECTED_ATTEMPTS = 4
# A forward line-search trial is accepted when its objective is within this
# fraction of max(|V|, 1) above the current one.
_SCOP_LINE_SEARCH_RELATIVE_TOLERANCE = 1.0e-8

# The outer step's two suppression holds (scasm-style), both read in effective
# degrees of freedom and both flatness tests. Write r_j = d log|S|+ / d rho_j,
# s_j = lambda_j tr(H^-1 S_j) (the EDF the penalty suppresses) and
# p_j = lambda_j beta' S_j beta / phi, so the working gradient is
#     g_j = p_j + s_j - r_j          (= 2 dV/d rho_j at fixed H).
# A component is held only where V is flat in the direction its step asks
# for, which is where Wood, Pya & Saefken (2016, outer step 4b) drop a
# smoothing parameter from the Newton update (V_rho ~ V_rho_rho ~ 0).
#
# Increase: as lambda_j grows the penalty absorbs its whole range and the
# residual EDF rEDF_j = r_j - s_j falls to zero, so an increase is held once
# rEDF_j < 0.05. An increase is asked for only when g_j < 0, and then
# |g_j| <= rEDF_j < 0.05: the bar bounds the slope it holds against.
#
# Decrease: as lambda_j -> 0 the slope tends to -r_j(0), where r_j(0) is the
# number of dimensions S_j adds to the other penalties' range (r_j is
# nondecreasing in rho_j, since d r_j / d rho_j = tr(A) - tr(A^2) >= 0 for
# A = lambda_j S^-1/2 S_j S^-1/2, whose eigenvalues lie in [0, 1]). For an
# isolated penalty r_j = rank(S_j) >= 1 at every lambda, so V -> +inf as
# lambda_j -> 0 and that end is never flat: a decrease is never held. Only
# where other penalties cover range(S_j) (a tensor-product margin) does r_j
# fall to zero, and the end flattens. A decrease is therefore held once
# r_j < 0.05 (which bounds s_j < r_j as well, Wood & Fasiolo 2017,
# Theorem 1) and only while the slope asking for it is itself under the bar,
# g_j < 0.05.
#
# Two earlier decrease bars were not flatness tests. tr(H^-1 S_j) < 0.05 is
# not an EDF (it carries units of 1/lambda and scales with S_j), and held
# every decrease once lambda_j > 20 rank(S_j): on the cleaned freMTPL2 book
# (VehAge, rank 8, lambda 468, s 5.4) the strict reference stopped there with
# a gradient of 0.79. s_j < 0.05 is an EDF, but for a strongly identified
# term (s_j at its optimum under 0.05) it held a band above the optimum where
# g_j reached 22, and fits started there stopped 'converged' at up to 3.2
# times the optimum's lambda.
_SCOP_SUPPRESSION_EDF = 0.05


def _scop_suppression_holds(
    prior_edf: float, suppressed_edf: float, penalty_edf: float
) -> tuple[bool, bool]:
    """Whether this component's increase and decrease are held at this iterate.

    ``prior_edf`` is ``r_j = d log|S|+ / d rho_j``, ``suppressed_edf`` is
    ``lambda_j tr(H^-1 S_j)`` and ``penalty_edf`` is
    ``lambda_j beta' S_j beta / phi``; see ``_SCOP_SUPPRESSION_EDF`` for both
    bars.
    """
    residual_edf = prior_edf - suppressed_edf
    gradient = penalty_edf + suppressed_edf - prior_edf
    increase_held = residual_edf < _SCOP_SUPPRESSION_EDF
    decrease_held = prior_edf < _SCOP_SUPPRESSION_EDF and gradient < _SCOP_SUPPRESSION_EDF
    return increase_held, decrease_held


@dataclass(frozen=True)
class _SCOPREMLFitContext:
    """Inputs shared by all coherent SCOP coefficient-mode evaluations."""

    dm: DesignMatrix
    distribution: Any
    link: Any
    groups: list[GroupSlice]
    y: NDArray
    sample_weight: NDArray
    offset_arr: NDArray
    pirls_tol: float
    max_pirls_iter: int
    reml_penalties: list[PenaltyComponent] | None
    convergence: str
    scop_joint: bool
    debug_recorder: Any
    likelihood_size: float
    weight_semantics: str
    gamma_scale_data: Any
    tweedie_scale_data: Any = None
    # Gaussian's weight-only saturated constant, which exists only under the
    # prior contract; defaulted so a hand-built context needs only the two
    # fields that decide a number.
    saturated_log_weight: float | None = None
    # The run's centring state across its coefficient fits (``SCOPRunCentring``):
    # an init field, so the retry contexts ``replace`` derives share it.
    scop_run_centring: SCOPRunCentring = field(
        default_factory=SCOPRunCentring, repr=False, compare=False
    )
    _penalty_context_cache: dict[int, _SCOPPenaltyContextEntry] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )


@dataclass(frozen=True)
class _SCOPREMLMode:
    """One fitted coefficient mode and its lambda-coherent LAML geometry."""

    lambdas: dict[str, float]
    result: Any
    xtwx: NDArray
    centered_xtwx: NDArray
    fisher_mean_x: NDArray
    fisher_sum_w: float
    scop_states: dict[int, dict]
    penalty: NDArray
    penalty_components: list[PenaltyComponent]
    joint_geometry: SCOPJointGeometry
    hessian_inverse: NDArray
    evaluation: Any
    log_det_h: float
    hessian_rank: int
    curvature_source: str
    mode_score: SCOPModeScore

    @property
    def objective(self) -> float:
        return float(self.evaluation.value)


def _multi_scop_discrete_cleanup_enabled(*, discrete: bool, scop_term_count: int) -> bool:
    return bool(discrete and scop_term_count > 1)


def _multi_scop_discrete_cleanup_names(
    *,
    estimated_names: set[str],
    scop_states: dict[int, dict],
    scop_term_count: int,
) -> set[str]:
    """Return the estimated SCOP names eligible for the discrete cleanup path."""
    if not scop_states:
        return set()

    all_scop_discrete = all(st.get("bin_idx") is not None for st in scop_states.values())
    if not _multi_scop_discrete_cleanup_enabled(
        discrete=all_scop_discrete,
        scop_term_count=scop_term_count,
    ):
        return set()

    eligible_names = {
        st["group_name"]
        for st in scop_states.values()
        if st.get("bin_idx") is not None and st["group_name"] in estimated_names
    }
    return eligible_names if len(eligible_names) > 1 else set()


def _get_scop_penalty_metadata(st: dict) -> tuple[float, float, NDArray]:
    """Return cached SCOP penalty metadata, computing it once if needed."""
    cached_rank = st.get("penalty_rank")
    cached_log_det = st.get("penalty_log_det_omega_plus")
    cached_eigvals = st.get("penalty_eigvals_omega")
    if cached_rank is not None and cached_log_det is not None and cached_eigvals is not None:
        eigvals = np.asarray(cached_eigvals, dtype=np.float64)
        rank = float(cached_rank)
        if (
            eigvals.ndim == 1
            and len(eigvals) == int(rank)
            and np.all(eigvals > 0.0)
            and np.isfinite(cached_log_det)
        ):
            return rank, float(cached_log_det), eigvals

    S_scop = st["S_scop"]
    eps_thresh = np.finfo(float).eps ** (2 / 3)
    eigvals = np.linalg.eigvalsh(S_scop)
    thresh = eps_thresh * max(eigvals.max(), 1e-12)
    rank = float(np.sum(eigvals > thresh))
    n_pos = int(rank)

    if n_pos > 0:
        sorted_eig = np.sort(eigvals)[::-1]
        pos_eigvals = np.asarray(sorted_eig[:n_pos], dtype=np.float64)
        log_det = float(np.sum(np.log(np.maximum(pos_eigvals, 1e-300))))
    else:
        pos_eigvals = np.array([], dtype=np.float64)
        log_det = 0.0

    st["penalty_rank"] = rank
    st["penalty_log_det_omega_plus"] = log_det
    st["penalty_eigvals_omega"] = pos_eigvals
    return rank, log_det, pos_eigvals


def _scop_jacobian_diag(st: dict) -> NDArray:
    """Return the diagonal of the solver-space coefficient-map Jacobian."""
    reparam = st.get("reparam")
    if reparam is not None and hasattr(reparam, "jacobian_diagonal"):
        beta_eff = np.asarray(st["beta_eff"], dtype=np.float64)
        return np.asarray(reparam.jacobian_diagonal(beta_eff), dtype=np.float64)
    gamma_eff = st.get("gamma_eff")
    if gamma_eff is not None:
        gamma_eff = np.asarray(gamma_eff, dtype=np.float64)
        if gamma_eff.ndim == 1 and np.all(np.isfinite(gamma_eff)):
            return gamma_eff
    beta_eff = np.asarray(st["beta_eff"], dtype=np.float64)
    return np.exp(np.clip(beta_eff, -500, 500))


def _update_multi_scop_discrete_stability_counts(
    *,
    lambdas_old: dict[str, float],
    lambdas_new: dict[str, float],
    active_names: set[str],
    stable_counts: dict[str, int],
) -> dict[str, int]:
    """Track generic per-name lambda stability across freeze and plateau signals.

    A name is counted as stable when it is either near the absolute lambda floor
    or moving by only a small log-step. The near-floor branch resets when a
    lambda first enters the floor region so freezing responds only to
    consecutive near-floor iterations.
    """
    updated = dict(stable_counts)
    floor_threshold = _MULTI_SCOP_DISCRETE_LAMBDA_FLOOR * _MULTI_SCOP_DISCRETE_FLOOR_FACTOR
    for name in active_names:
        lam_old = max(lambdas_old[name], 1.0e-10)
        lam_new = max(lambdas_new[name], 1.0e-10)
        log_step = abs(np.log(lam_new) - np.log(lam_old))
        near_floor_old = lam_old <= floor_threshold
        near_floor_new = lam_new <= floor_threshold
        if near_floor_new:
            updated[name] = updated.get(name, 0) + 1 if near_floor_old else 1
        elif log_step < _MULTI_SCOP_DISCRETE_LOG_STEP_TOL:
            updated[name] = updated.get(name, 0) + 1
        else:
            updated[name] = 0
    return updated


def _freeze_multi_scop_discrete_lambdas(
    *,
    active_names: set[str],
    frozen_names: set[str],
    lambdas_new: dict[str, float],
    stable_counts: dict[str, int],
) -> tuple[set[str], set[str]]:
    """Freeze only floor-pinned names once the generic stability counter matures."""
    active_out = set(active_names)
    frozen_out = set(frozen_names)
    for name in list(active_names):
        lam_new = lambdas_new[name]
        near_floor = lam_new <= (
            _MULTI_SCOP_DISCRETE_LAMBDA_FLOOR * _MULTI_SCOP_DISCRETE_FLOOR_FACTOR
        )
        stable_long_enough = stable_counts.get(name, 0) >= _MULTI_SCOP_DISCRETE_MIN_STABLE_ITERS
        if near_floor and stable_long_enough:
            active_out.discard(name)
            frozen_out.add(name)
    return active_out, frozen_out


def _multi_scop_discrete_plateau_converged(
    *,
    obj_rel_change: float,
    lambdas_old: dict[str, float],
    lambdas_new: dict[str, float],
    active_names: set[str],
) -> bool:
    """Require objective flatness, then check active-set log-step stability."""
    if not active_names:
        return obj_rel_change < _MULTI_SCOP_DISCRETE_OBJ_REL_TOL
    active_changes = [
        abs(np.log(max(lambdas_new[name], 1.0e-10)) - np.log(max(lambdas_old[name], 1.0e-10)))
        for name in active_names
    ]
    max_active_change = max(active_changes) if active_changes else 0.0
    return (
        obj_rel_change < _MULTI_SCOP_DISCRETE_OBJ_REL_TOL
        and max_active_change < _MULTI_SCOP_DISCRETE_ACTIVE_PLATEAU_TOL
    )


@dataclass(frozen=True)
class _SCOPPenaltyContextEntry:
    """One authenticated latent target, independent of coefficients and Jacobians."""

    component: PenaltyComponent
    source_key: tuple
    reparam: object
    descriptor: tuple[tuple[str, object], ...]
    geometry: object

    def matches(self, group_index: int, state: dict) -> bool:
        matrix = np.asarray(state["S_scop"])
        if (
            self.source_key != _scop_penalty_source_key(group_index, state, matrix)
            or self.reparam is not state.get("reparam")
            or _context_geometry([self.component]) is not self.geometry
        ):
            return False
        for name, expected in self.descriptor:
            actual = getattr(self.component, name)
            if isinstance(expected, np.ndarray):
                if actual is not expected or actual.flags.writeable:
                    return False
            elif type(actual) is not type(expected) or actual != expected:
                return False
        return bool(
            np.array_equal(matrix, self.component.omega_ssp)
            and state.get("penalty_rank") == self.component.rank
            and state.get("penalty_log_det_omega_plus") == self.component.log_det_omega_plus
            and np.array_equal(state.get("penalty_eigvals_omega"), self.component.eigvals_omega)
        )


def _scop_penalty_source_key(group_index: int, state: dict, matrix: NDArray) -> tuple:
    return (
        group_index,
        state["group_name"],
        state["group_sl"],
        matrix.shape,
        matrix.dtype.str,
    )


def build_scop_penalty_components(
    scop_states: dict[int, dict],
    *,
    _cache: dict[int, _SCOPPenaltyContextEntry] | None = None,
    _lambdas: dict[str, float] | None = None,
) -> list[PenaltyComponent]:
    """Build PenaltyComponent objects for SCOP terms.

    For SCOP terms, omega_ssp = S_scop (first-diff penalty in beta_eff space).
    No R_inv transform -- SCOP bypasses SSP reparameterization.

    The optimizer's private cache retains the exact fixed latent target and
    its checked unit-weight summary. It owns no beta, Jacobian or fitted
    Hessian. Standalone callers keep the uncached descriptor contract.

    Parameters
    ----------
    scop_states : dict
        Keyed by group index. Each value has keys:
        "S_scop", "group_sl", "group_name", "beta_eff".

    Returns
    -------
    list[PenaltyComponent]
    """
    components = []

    for gi, st in scop_states.items():
        S_scop = st["S_scop"]
        if _cache is not None:
            cached = _cache.get(gi)
            if cached is not None and cached.matches(gi, st):
                components.append(cached.component)
                continue
            matrix = np.asarray(S_scop)
            group_slice = st["group_sl"]
            if (
                isinstance(gi, bool)
                or not isinstance(gi, int | np.integer)
                or gi < 0
                or not isinstance(st["group_name"], str)
                or not st["group_name"]
                or not isinstance(group_slice, slice)
                or group_slice.step not in (None, 1)
                or not isinstance(group_slice.start, int | np.integer)
                or not isinstance(group_slice.stop, int | np.integer)
                or group_slice.start < 0
                or matrix.shape != (group_slice.stop - group_slice.start,) * 2
            ):
                raise ValueError("SCOP penalty context has invalid local geometry")
            # A changed matrix or descriptor must not inherit stale spectral
            # metadata. Stage it locally so a refusal preserves the last
            # authenticated cache entry and the source's previous evidence.
            metadata_state = {
                key: value
                for key, value in st.items()
                if key
                not in {"penalty_rank", "penalty_log_det_omega_plus", "penalty_eigvals_omega"}
            }
        else:
            metadata_state = st
        rank, log_det, pos_eigvals = _get_scop_penalty_metadata(metadata_state)

        pc = PenaltyComponent(
            name=st["group_name"],
            group_name=st["group_name"],
            group_index=gi,
            group_sl=st["group_sl"],
            omega_raw=S_scop,
            omega_ssp=S_scop,
            rank=rank,
            log_det_omega_plus=log_det,
            eigvals_omega=pos_eigvals,
        )
        if _cache is not None:
            _attach_context_geometry([pc])
            pc.omega_raw = pc.omega_ssp
            pc.eigvals_omega = _frozen_array(pc.eigvals_omega)
            geometry = _context_geometry([pc])
            if geometry is None:
                raise ValueError("SCOP penalty context could not bind its local target")
            # Preserve the existing inactive-weight shortcut. Invalid weights
            # remain for the established consumer to reject; a new zero-only
            # target must not request an unused unit-weight certificate.
            value = 1.0 if _lambdas is None else _lambdas.get(pc.name, 1.0)
            try:
                active = np.isfinite(float(value)) and float(value) > 0.0
            except (TypeError, ValueError, OverflowError):
                active = False
            if active:
                geometry.evaluate(np.ones(1))
                entry = _SCOPPenaltyContextEntry(
                    component=pc,
                    source_key=_scop_penalty_source_key(gi, st, matrix),
                    reparam=st.get("reparam"),
                    descriptor=tuple((item.name, getattr(pc, item.name)) for item in fields(pc)),
                    geometry=geometry,
                )
                st["penalty_rank"] = rank
                st["penalty_log_det_omega_plus"] = log_det
                st["penalty_eigvals_omega"] = pc.eigvals_omega
                _cache[gi] = entry
        components.append(pc)

    return components


def _merge_scop_penalty_components(
    base_components: list[PenaltyComponent] | None,
    scop_components: list[PenaltyComponent],
) -> list[PenaltyComponent]:
    """Replace mapped-coordinate components for SCOP-owned coefficient blocks."""
    scop_group_indices = {component.group_index for component in scop_components}
    ordinary = [
        component
        for component in (base_components or [])
        if component.group_index not in scop_group_indices
    ]
    return ordinary + scop_components


def compute_scop_aware_penalty_quad(
    result_beta: NDArray,
    S: NDArray,
    scop_states: dict[int, dict],
    lambdas: dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> float:
    """Compute penalty quadratic with correct SCOP beta_eff contributions.

    For non-SCOP groups, result.beta @ S @ result.beta is correct (SSP space).
    For SCOP groups, ``result.beta`` contains the elementwise mapped
    ``gamma_eff = forward(beta_eff)``, but the penalty is defined on
    ``beta_eff``: ``lambda * beta_eff^T @ S_scop @ beta_eff``.
    We subtract the wrong gamma-space contribution and add the correct
    beta_eff-space contribution.

    Parameters
    ----------
    result_beta : (p,) coefficient vector (contains gamma for SCOP groups)
    S : (p, p) full penalty matrix (block-diagonal, includes lambda * S_scop blocks)
    scop_states : SCOP converged state dict
    lambdas : dict of lambda values keyed by component name
    """
    if not scop_states:
        return float(result_beta @ S @ result_beta)

    pq = float(result_beta @ S @ result_beta)

    for gi, st in scop_states.items():
        sl = st["group_sl"]
        beta_eff = st["beta_eff"]
        gamma_eff = result_beta[sl]

        # Remove the complete mapped-coordinate block, including every named
        # component that may overlap this SCOP group.
        pq -= float(gamma_eff @ S[sl, sl] @ gamma_eff)
        matching = (
            [component for component in reml_penalties if component.group_index == gi]
            if reml_penalties is not None
            else []
        )
        if matching:
            for component in matching:
                pq += lambdas[component.name] * penalty_component_quadratic(component, beta_eff)
        else:
            lam = lambdas.get(st["group_name"], 0.0)
            pq += lam * float(beta_eff @ st["S_scop"] @ beta_eff)

    return pq


def assemble_joint_hessian(
    XtWX_plus_S: NDArray,
    scop_states: dict[int, dict],
    *,
    XtW1: NDArray | None = None,
    sum_W: float | None = None,
) -> tuple[NDArray, dict[str, slice]]:
    """Assemble the intercept-profiled joint Hessian in latent coordinates.

    The XtWX_plus_S matrix is in gamma space for SCOP groups: the SCOP
    diagonal block has only ``lambda * S_scop`` (missing data curvature),
    and the cross-blocks ``X_linear^T W B_scop`` lack the SCOP Jacobian
    factor.

    This function:

    1. Replaces each SCOP diagonal block with ``H_scop_penalized`` (the
       full Newton Hessian in beta_eff space, including data curvature).
    2. Transforms cross-blocks to beta_eff space by scaling columns
       (for ``H[other, scop]``) and rows (for ``H[scop, other]``) by
       the diagonal of the SCOP Jacobian
       ``d(gamma_eff)/d(beta_eff)``.

    Parameters
    ----------
    XtWX_plus_S : (p, p) ndarray
        The linear-system penalized Gram matrix.
    scop_states : dict
        SCOP converged state dict, keyed by group index. Each value
        must contain "group_sl", "H_scop_penalized", "group_name",
        and "beta_eff".

    Returns
    -------
    H_joint : (p, p) ndarray
        Joint Hessian with SCOP blocks and cross-blocks in beta_eff space.
    mapping : dict
        Maps group_name to the slice in H_joint for each SCOP group.
    """
    if (XtW1 is None) != (sum_W is None):
        raise ValueError("XtW1 and sum_W must be provided together")
    if not scop_states and XtW1 is None:
        return XtWX_plus_S, {}

    p = XtWX_plus_S.shape[0]
    H_joint = XtWX_plus_S.copy()
    mapping = {}

    # Collect all SCOP indices so we can identify "other" indices
    scop_slices = []
    for gi, st in scop_states.items():
        scop_slices.append(st["group_sl"])

    all_scop_idx = (
        np.concatenate([np.arange(sl.start, sl.stop) for sl in scop_slices])
        if scop_slices
        else np.empty(0, dtype=int)
    )
    other_idx = np.setdiff1d(np.arange(p), all_scop_idx)

    for gi, st in scop_states.items():
        sl = st["group_sl"]
        H_scop = st["H_scop_penalized"]
        name = st["group_name"]
        j_diag = _scop_jacobian_diag(st)

        # Replace diagonal SCOP block with full Newton Hessian
        H_joint[sl, sl] = H_scop
        mapping[name] = sl

        # Transform cross-blocks: gamma-space → beta_eff-space
        # H[other, scop] = X_other^T W B_scop  →  scale columns by j_diag
        if other_idx.size > 0:
            scop_idx = np.arange(sl.start, sl.stop)
            H_joint[np.ix_(other_idx, scop_idx)] *= j_diag[np.newaxis, :]
            H_joint[np.ix_(scop_idx, other_idx)] *= j_diag[:, np.newaxis]

    # Transform SCOP-SCOP cross-blocks: H_ij(beta_eff) = diag(j_i) @ H_ij(gamma) @ diag(j_j)
    scop_items = list(scop_states.items())
    for idx_a in range(len(scop_items)):
        gi_a, st_a = scop_items[idx_a]
        sl_a = st_a["group_sl"]
        j_a = _scop_jacobian_diag(st_a)
        for idx_b in range(idx_a + 1, len(scop_items)):
            gi_b, st_b = scop_items[idx_b]
            sl_b = st_b["group_sl"]
            j_b = _scop_jacobian_diag(st_b)
            idx_a_arr = np.arange(sl_a.start, sl_a.stop)
            idx_b_arr = np.arange(sl_b.start, sl_b.stop)
            H_joint[np.ix_(idx_a_arr, idx_b_arr)] *= j_a[:, np.newaxis] * j_b[np.newaxis, :]
            H_joint[np.ix_(idx_b_arr, idx_a_arr)] *= j_b[:, np.newaxis] * j_a[np.newaxis, :]

    if XtW1 is not None:
        assert sum_W is not None
        intercept_cross = np.asarray(XtW1, dtype=np.float64)
        if intercept_cross.shape != (p,):
            raise ValueError("XtW1 must match the slope coefficient dimension")
        if not np.all(np.isfinite(intercept_cross)):
            raise ValueError("XtW1 must be finite")
        if not np.isfinite(sum_W) or sum_W <= 0.0:
            raise ValueError("sum_W must be positive and finite")
        intercept_cross = intercept_cross.copy()
        for st in scop_states.values():
            sl = st["group_sl"]
            intercept_cross[sl] *= _scop_jacobian_diag(st)
        H_joint -= np.outer(intercept_cross, intercept_cross) / sum_W

    # Restrict to the range the SCOP steps could resolve. The diagonal blocks
    # arrive already restricted, but the cross-blocks assembled above still
    # couple other coefficients to a direction the solver froze, and that
    # leakage is enough to leave the joint matrix indefinite where consumers
    # decompose it. Projecting the assembled matrix covers both. A fit whose
    # steps discarded nothing is returned untouched.
    H_joint = restrict_to_scop_resolved_range(H_joint, scop_states)

    return H_joint, mapping


def _result_intercept_moments(
    result: Any,
    *,
    width: int,
    fisher_mean_x: NDArray | None = None,
    fisher_sum_w: float | None = None,
) -> tuple[NDArray, float]:
    """Recover intercept moments from the working cache or compatibility metadata."""
    if (fisher_mean_x is None) != (fisher_sum_w is None):
        raise ValueError("fisher_mean_x and fisher_sum_w must be provided together")
    if fisher_mean_x is not None:
        assert fisher_sum_w is not None
        mean_x = np.asarray(fisher_mean_x, dtype=np.float64)
        sum_w = float(fisher_sum_w)
        if mean_x.shape != (width,):
            raise RuntimeError("SCOP REML intercept geometry has the wrong width")
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            raise RuntimeError("SCOP REML intercept weight sum must be positive and finite")
        return sum_w * mean_x, sum_w

    rank_info = result.rank_info
    if rank_info is None:
        raise RuntimeError("SCOP REML requires retained intercept geometry")
    sum_w = float(rank_info.sum_w)
    mean_x = np.asarray(rank_info.mean_x, dtype=np.float64)
    if mean_x.shape != (width,):
        raise RuntimeError("SCOP REML intercept geometry has the wrong width")
    return sum_w * mean_x, sum_w


def _reml_evaluation_phi(
    evaluation: Any,
    *,
    scale_known: bool,
    fallback_likelihood_size: float,
) -> float:
    """Return the scale paired with one authoritative REML evaluation."""
    if scale_known:
        return 1.0
    if evaluation.profiled_scale is not None:
        return float(evaluation.profiled_scale.phi)
    penalty_nullity = float(evaluation.penalty_nullity or 0.0)
    return max(
        float(evaluation.penalized_deviance)
        / max(float(fallback_likelihood_size) - penalty_nullity, 1.0),
        1.0e-10,
    )


def _evaluate_scop_reml_mode(
    context: _SCOPREMLFitContext,
    lambdas: dict[str, float],
    *,
    result: Any,
    xtwx: NDArray,
    centered_xtwx: NDArray,
    fisher_mean_x: NDArray,
    fisher_sum_w: float | None = None,
    scop_states: dict[int, dict],
    penalty_components: list[PenaltyComponent] | None = None,
    penalty: NDArray | None = None,
    mode_score: SCOPModeScore | None = None,
    eta_unclipped: NDArray | None = None,
) -> _SCOPREMLMode:
    """Assemble and evaluate LAML from one lambda-coherent fitted mode."""
    if penalty_components is None:
        penalty_components = _merge_scop_penalty_components(
            context.reml_penalties,
            build_scop_penalty_components(
                scop_states, _cache=context._penalty_context_cache, _lambdas=lambdas
            ),
        )
    if penalty is None:
        penalty = build_penalty_matrix(
            list(context.dm.group_matrices),
            context.groups,
            lambdas,
            context.dm.p,
            reml_penalties=penalty_components,
        )
    xtw1, sum_w = _result_intercept_moments(
        result,
        width=context.dm.p,
        fisher_mean_x=fisher_mean_x if fisher_sum_w is not None else None,
        fisher_sum_w=fisher_sum_w,
    )
    if mode_score is None:
        try:
            mode_score = scop_penalized_mode_score(
                dm=context.dm,
                distribution=context.distribution,
                link=context.link,
                y=context.y,
                sample_weight=context.sample_weight,
                offset_arr=context.offset_arr,
                result=result,
                latent_penalty=penalty,
                scop_states=scop_states,
                centered_fisher_gram=centered_xtwx,
                fisher_mean_x=fisher_mean_x,
                fisher_sum_w=sum_w,
                eta_unclipped=eta_unclipped,
            )
        except ObservedGeometryInfeasibleError as exc:
            # An unscoreable mode is one more infeasible point to a power
            # search, not a dead search. Left untyped it escapes as a bare
            # ValueError past every `except ObservedModeNotCertifiedError`
            # handler, because that family is RuntimeError-derived.
            raise ObservedModeNotConvergedError(
                f"SCOP could not score its penalized coefficient mode at this point: {exc}",
                infeasible_detail="SCOP mode score refused the penalized mode",
            ) from exc

    def terminal_fisher_weights() -> NDArray:
        eta_raw = (
            context.dm.matvec(result.beta) + result.intercept + context.offset_arr
            if eta_unclipped is None
            else np.asarray(eta_unclipped, dtype=np.float64)
        )
        eta = stabilize_eta(eta_raw, context.link)
        mu = clip_mu(context.link.inverse(eta), context.distribution)
        return fisher_working_weights(
            distribution=context.distribution,
            link=context.link,
            mu=mu,
            eta=eta,
            sample_weight=context.sample_weight,
        )

    curvature = (
        classify_scop_reml_curvature(context.distribution, context.link)
        if scop_states
        else "fisher"
    )
    if curvature == "observed":
        joint_geometry = build_observed_scop_joint_geometry(
            dm=context.dm,
            distribution=context.distribution,
            link=context.link,
            y=context.y,
            sample_weight=context.sample_weight,
            offset_arr=context.offset_arr,
            result=result,
            penalty=penalty,
            scop_states=scop_states,
            fisher_XtWX=xtwx,
            fisher_XtW1=xtw1,
            fisher_sum_W=sum_w,
            centered_fisher_gram=centered_xtwx,
            fisher_mean_x=fisher_mean_x,
            eta_unclipped=eta_unclipped,
        )
        hessian_inverse = joint_geometry.hessian_inverse
        log_det_h = joint_geometry.log_det_H
        hessian_rank = joint_geometry.hessian_rank
        curvature_source = joint_geometry.curvature_source
    else:
        joint_geometry = build_cached_scop_joint_geometry(
            raw_fisher_gram=xtwx,
            fisher_xtw=xtw1,
            fisher_sum_w=sum_w,
            latent_penalty=penalty,
            scop_states=scop_states,
            centered_fisher_gram=centered_xtwx,
            fisher_mean_x=fisher_mean_x,
            dm=context.dm,
            fisher_weights=terminal_fisher_weights,
        )
        hessian_inverse = joint_geometry.hessian_inverse
        log_det_h = joint_geometry.log_det_H
        hessian_rank = joint_geometry.hessian_rank
        curvature_source = joint_geometry.curvature_source
    evaluation = reml_laml_objective(
        context.dm,
        context.distribution,
        context.link,
        context.groups,
        context.y,
        result,
        lambdas,
        context.sample_weight,
        context.offset_arr,
        XtWX=xtwx,
        XtW1=xtw1,
        sum_W=sum_w,
        log_det_H=log_det_h,
        hessian_rank=hessian_rank,
        S_override=penalty,
        reml_penalties=penalty_components,
        scop_states=scop_states,
        likelihood_size=context.likelihood_size,
        saturated_log_weight=context.saturated_log_weight,
        weight_semantics=context.weight_semantics,
        gamma_scale_data=context.gamma_scale_data,
        tweedie_scale_data=context.tweedie_scale_data,
        return_evaluation=True,
    )
    return _SCOPREMLMode(
        lambdas=lambdas.copy(),
        result=result,
        xtwx=xtwx,
        centered_xtwx=centered_xtwx,
        fisher_mean_x=fisher_mean_x,
        fisher_sum_w=sum_w,
        scop_states=scop_states,
        penalty=penalty,
        penalty_components=penalty_components,
        joint_geometry=joint_geometry,
        hessian_inverse=hessian_inverse,
        evaluation=evaluation,
        log_det_h=log_det_h,
        hessian_rank=hessian_rank,
        curvature_source=curvature_source,
        mode_score=mode_score,
    )


@dataclass(frozen=True)
class _SCOPModeNewtonCorrection:
    """The certificate's joint Newton correction in latent coordinates.

    ``latent_beta`` holds the ordinary coefficients with each SCOP block's
    latent ``beta_eff`` in place of its mapped coefficients; ``slope`` and
    ``intercept`` are the profiled Newton step on the estimable range that
    certification measures.
    """

    latent_beta: NDArray
    slope: NDArray
    intercept: float
    relative: float


def _scop_mode_newton_relative(mode: _SCOPREMLMode) -> float:
    """Return the estimable-range Newton correction for mode certification."""
    return _scop_mode_newton_correction(mode).relative


def _scop_mode_newton_correction(mode: _SCOPREMLMode) -> _SCOPModeNewtonCorrection:
    """Return the estimable-range Newton correction for mode certification.

    Componentwise relative scores are deliberately retained as a diagnostic,
    but they are not a sufficient convergence test at a flat SCOP boundary.
    There the exponential Jacobian and the exact penalty score both vanish,
    so harmless penalty-matvec noise can have an order-one *relative* score.
    The factor-certified pseudoinverse instead measures the coefficient
    correction on the estimable range, as required for the rank-deficient
    latent geometry described by Pya and Wood.
    """
    geometry = mode.joint_geometry
    score = mode.mode_score
    latent_beta = np.asarray(mode.result.beta, dtype=np.float64).copy()
    fisher_transformed_mean = np.asarray(mode.fisher_mean_x, dtype=np.float64).copy()
    for state in mode.scop_states.values():
        group_slice = state["group_sl"]
        latent_beta[group_slice] = np.asarray(state["beta_eff"], dtype=np.float64)
        fisher_transformed_mean[group_slice] *= _scop_jacobian_diag(state)

    geometry_mean = geometry.transformed_mean_x
    if geometry_mean is None:
        geometry_mean = geometry.transformed_intercept_cross / geometry.sum_w
    geometry_mean = np.asarray(geometry_mean, dtype=np.float64)

    # ``score.slopes`` is profiled with the retained Fisher mean. Reconstruct
    # the raw slope score, then profile it with the curvature actually used by
    # this mode (observed or Fisher) before applying that same pseudoinverse.
    raw_slope_score = score.slopes + fisher_transformed_mean * score.intercept
    profiled_score = raw_slope_score - geometry_mean * score.intercept
    # Stationarity is only meaningful where the solver could move. The SCOP
    # Newton step truncates its augmented factor at ``sqrt(eps)``, and the
    # discarded directions are ones the data cannot resolve -- no iteration
    # drives their score to zero, so requiring it here would reject every
    # boundary mode. This geometry's own estimable range is certified over the
    # whole centered model at its own scale, so it does not coincide with the
    # solver's; the solver's is the one the coefficients actually obey.
    #
    # The score is projected *before* the pseudoinverse, not the correction
    # after it: ``hessian_inverse`` is not diagonal in this basis, so a score
    # component along a discarded direction re-emerges as a correction pointing
    # somewhere else entirely, which no projection of the result can remove.
    resolved = scop_resolved_range_projector(mode.scop_states, len(profiled_score))
    if resolved is not None:
        profiled_score = resolved @ profiled_score

    slope_correction = geometry.hessian_inverse @ profiled_score
    if resolved is not None:
        # The correction must also lie where the solver can move.
        slope_correction = resolved @ slope_correction

    intercept_correction = (
        score.intercept - float(geometry.transformed_intercept_cross @ slope_correction)
    ) / geometry.sum_w

    slope_relative = float(
        np.max(
            np.abs(slope_correction) / np.maximum(1.0, np.abs(latent_beta)),
            initial=0.0,
        )
    )
    intercept_relative = abs(intercept_correction) / max(
        1.0,
        abs(float(mode.result.intercept)),
    )
    return _SCOPModeNewtonCorrection(
        latent_beta=latent_beta,
        slope=slope_correction,
        intercept=float(intercept_correction),
        relative=max(slope_relative, intercept_relative),
    )


def _newton_polished_warm_start(
    mode: _SCOPREMLMode, correction: _SCOPModeNewtonCorrection
) -> tuple[NDArray, float, dict[int, dict]] | None:
    """Warm start for a certification retry: the failed mode plus its Newton step.

    The inner SCOP solve alternates an ordinary-block solve with a SCOP Newton
    step, a block-coordinate iteration that converges only linearly: from a
    mode that met its coefficient-step test but not the certificate, the
    retry took about ten more inner iterations, each a full Gram build. The
    certificate has already computed the joint Newton step over every
    coefficient (Pya & Wood fit SCAM coefficients by full Newton), so the
    retry starts from that step instead. The retry is still an ordinary
    converged inner fit at the tightened tolerance, and its mode is certified
    afresh: this changes only where it starts, never what it must satisfy.
    Returns None when the step is not finite, so the caller keeps the plain
    warm start.
    """
    polished = correction.latent_beta + correction.slope
    intercept = float(mode.result.intercept) + correction.intercept
    if not (np.all(np.isfinite(polished)) and np.isfinite(intercept)):
        return None
    beta = polished.copy()
    states: dict[int, dict] = {}
    for group_index, state in mode.scop_states.items():
        group_slice = state["group_sl"]
        beta_eff = polished[group_slice].copy()
        beta[group_slice] = state["reparam"].forward(beta_eff)
        states[group_index] = {**state, "beta_eff": beta_eff}
    if not np.all(np.isfinite(beta)):
        return None
    return beta, intercept, states


def _scop_mode_tolerance(mode: _SCOPREMLMode) -> float:
    """Return the rank-aware numerical floor for terminal mode certification.

    The authoritative observation-factor policy resolves directions only to
    ``sqrt(eps)`` relative accuracy. A joint rank-dimensional correction
    aggregates that factor/score roundoff at root-rank scale. This remains a
    numerical floor, not an alternative score-based convergence criterion.

    Deliberately independent of the solver tolerance: certification asks
    whether the mode is stationary to the accuracy the factor policy can
    resolve at all, which no request for a looser inner fit can relax.  The
    ``pirls_tol`` term this once carried was dead by arithmetic --
    ``10 * min(pirls_tol, 1e-10) <= 1e-9`` can never exceed this floor,
    which is at least ``sqrt(eps) ~ 1.49e-8`` -- so the bar measured as
    exactly this value on every certification the suites perform.

    Do not generalise that deletion to the sibling bars in
    ``model/reml_finalize.py`` (the same clamped term) and ``reml/direct.py``
    (the clamp applies only on the observed-geometry path; the Fisher path
    is unclamped ``10 * pirls_tol``): both sit over a ``100*eps`` floor,
    far below ``1e-9``, so there the tolerance arm is live and load-bearing.
    """
    return float(np.sqrt(max(1, mode.hessian_rank) * np.finfo(np.float64).eps))


def _scop_certification_failure(
    mode_newton_relative: float, mode_tolerance: float, componentwise_score: float
) -> ObservedModeNotCertifiedError:
    """The typed certification failure, carrying the metric that FAILED.

    The failing condition is ``mode_newton_relative > mode_tolerance``, and
    these metrics are deliberately distinct: an ill-conditioned mode can
    hold a sub-threshold componentwise score with an excessive Newton
    correction. Reporting the componentwise score as the achieved value
    made the infeasibility reason and ``PublicationModeError`` claim a
    score that never exceeded the bar. Typed like the observed-geometry
    gates: to a power search this is "no usable penalized mode at this
    point", an infeasible power to route around, not a dead search; the
    SCOP-specific detail rides in the hint.
    """
    return ObservedModeNotCertifiedError(
        mode_newton_relative,
        mode_tolerance,
        hint=(
            "SCOP latent mode certification: the relative Newton correction "
            f"exceeded the bar; componentwise mode score={componentwise_score:.3g}."
        ),
    )


def _fit_scop_reml_mode(
    context: _SCOPREMLFitContext,
    lambdas: dict[str, float],
    *,
    beta_init: NDArray | None,
    intercept_init: float | None,
    scop_state_init: dict[int, dict] | None,
    phase: str,
    reml_iteration: int,
    line_search_iteration: int | None = None,
    trial_alpha: float | None = None,
    require_converged: bool,
    _certification_retry: int = 0,
) -> _SCOPREMLMode | None:
    """Fit and evaluate one mode, optionally rejecting a failed inner solve."""
    debug_context: dict[str, Any] = {
        "phase": phase,
        "reml_iteration": reml_iteration,
    }
    if line_search_iteration is not None:
        debug_context["line_search_iteration"] = line_search_iteration
    if trial_alpha is not None:
        debug_context["trial_alpha"] = float(trial_alpha)

    trace_run = getattr(context.debug_recorder, "trace_run", None)
    trace_purpose = {
        "bootstrap": "reml_bootstrap",
        "candidate": "reml_candidate",
        "reml": "reml_candidate",
        "line_search": "reml_line_search",
        "final": "reml_final",
        "fixed": "reml_fixed",
    }.get(phase, f"reml_{phase}")
    working_cache: dict[str, Any] = {}
    inner_tol = (
        min(context.pirls_tol, 1.0e-10)
        if any(group.monotone_engine == "scop" for group in context.groups)
        and classify_scop_reml_curvature(context.distribution, context.link) == "observed"
        else context.pirls_tol
    )
    irls_out: Any = fit_irls_direct(
        X=context.dm,
        y=context.y,
        weights=context.sample_weight,
        family=context.distribution,
        link=context.link,
        groups=context.groups,
        lambda2=lambdas,
        offset=context.offset_arr,
        beta_init=beta_init,
        intercept_init=intercept_init,
        tol=inner_tol,
        max_iter=context.max_pirls_iter,
        return_xtwx=True,
        return_scop_state=True,
        reml_penalties=context.reml_penalties,
        # A LAML evaluation requires a coefficient mode. Deviance plateaus can
        # precede latent SCOP stationarity, especially near an interpolating
        # fit where the penalty deliberately trades a tiny deviance increase
        # for a much smaller quadratic.
        convergence="coefficients",
        _scop_joint=context.scop_joint,
        scop_state_init=scop_state_init,
        debug_recorder=context.debug_recorder,
        debug_context=debug_context,
        trace_run=trace_run,
        trace_purpose=trace_purpose,
        _compute_scop_postfit_inference=False,
        compute_rank_info=False,
        _compute_fit_statistics=False,
        _compute_reml_geometry=False,
        cache_out=working_cache,
        weight_semantics=context.weight_semantics,
        _scop_run_centring=context.scop_run_centring,
    )
    scop_states: dict[int, dict]
    if len(irls_out) == 4:
        result, _, xtwx, scop_states = irls_out
    else:
        result, _, xtwx = irls_out
        scop_states = {}

    if require_converged and not result.converged:
        # getattr, unlike the load-bearing predicate in observed_geometry: this is
        # a best-effort note on a path that is already failing, and it must not
        # be the thing that raises.
        if getattr(result, "termination_reason", None) == "max_iter":
            # Budget exhaustion is refused here by design: item 2c retired the
            # rule that specially accepted a non-converged inner fit, because
            # PR #176 removed its cause rather than working around it. But under
            # observed curvature ``inner_tol`` is a FIXED ceiling, not scaled to
            # the problem, and a step-length test cannot fire when the
            # iteration's round-off floor sits above it -- the ordinary REML
            # path was measured failing exactly that way, 9x to 646x above the
            # same ceiling. No shape-constrained fit reaching it has been
            # produced (measured: 922 inner fits at this tolerance, none
            # exhausted), so the gate stands. Say so on the way out rather than
            # let a floor-limited fit read as a fit that would not settle.
            logger.warning(
                "SCOP inner fit exhausted %d PIRLS iterations at tol=%.1e without "
                "meeting its step-length test, and is refused. If that tolerance "
                "sits below this problem's round-off floor then the mode may in "
                "fact have been reached and the test simply cannot fire; compare "
                "the achieved coefficient step against the tolerance before "
                "reading this as a fit that would not settle.",
                int(getattr(result, "n_iter", -1)),
                inner_tol,
            )
        return None
    rank_info = result.rank_info
    cached_mean_x = working_cache.get("mean_x")
    cached_sum_w = working_cache.get("sum_W")
    if cached_mean_x is None or cached_sum_w is None:
        if rank_info is None:
            raise RuntimeError("SCOP REML requires retained centered fit geometry")
        cached_mean_x = rank_info.mean_x
        cached_sum_w = rank_info.sum_w
    fisher_mean_x = np.asarray(cached_mean_x, dtype=np.float64)
    fisher_sum_w = float(cached_sum_w)
    if fisher_mean_x.shape != (context.dm.p,):
        raise RuntimeError("SCOP REML centered mean has the wrong width")
    if not np.isfinite(fisher_sum_w) or fisher_sum_w <= 0.0:
        raise RuntimeError("SCOP REML centered weight sum must be positive and finite")
    centered_xtwx = working_cache.get("centered_XtWX")
    if centered_xtwx is None:
        # Compatibility for injected/custom solvers. Production direct IRLS
        # always publishes the stable centered matrix through ``cache_out``.
        centered_xtwx = xtwx - fisher_sum_w * np.outer(
            fisher_mean_x,
            fisher_mean_x,
        )
    centered_xtwx = np.asarray(centered_xtwx, dtype=np.float64)
    penalty_components = _merge_scop_penalty_components(
        context.reml_penalties,
        build_scop_penalty_components(
            scop_states, _cache=context._penalty_context_cache, _lambdas=lambdas
        ),
    )
    penalty = build_penalty_matrix(
        list(context.dm.group_matrices),
        context.groups,
        lambdas,
        context.dm.p,
        reml_penalties=penalty_components,
    )
    sum_w = fisher_sum_w
    if scop_states:
        try:
            mode_score = scop_penalized_mode_score(
                dm=context.dm,
                distribution=context.distribution,
                link=context.link,
                y=context.y,
                sample_weight=context.sample_weight,
                offset_arr=context.offset_arr,
                result=result,
                latent_penalty=penalty,
                scop_states=scop_states,
                centered_fisher_gram=centered_xtwx,
                fisher_mean_x=fisher_mean_x,
                fisher_sum_w=sum_w,
                eta_unclipped=working_cache.get("eta_unclipped"),
            )
        except ObservedGeometryInfeasibleError as exc:
            # An unscoreable mode is one more infeasible point to a power
            # search, not a dead search. Left untyped it escapes as a bare
            # ValueError past every `except ObservedModeNotCertifiedError`
            # handler, because that family is RuntimeError-derived.
            raise ObservedModeNotConvergedError(
                f"SCOP could not score its penalized coefficient mode at this point: {exc}",
                infeasible_detail="SCOP mode score refused the penalized mode",
            ) from exc
    else:
        mode_score = SCOPModeScore(
            intercept=0.0,
            slopes=np.zeros(context.dm.p, dtype=np.float64),
            max_abs=0.0,
            relative_max=0.0,
        )
    mode = _evaluate_scop_reml_mode(
        context,
        lambdas,
        result=result,
        xtwx=xtwx,
        centered_xtwx=centered_xtwx,
        fisher_mean_x=fisher_mean_x,
        fisher_sum_w=fisher_sum_w,
        scop_states=scop_states,
        penalty_components=penalty_components,
        penalty=penalty,
        mode_score=mode_score,
        eta_unclipped=working_cache.get("eta_unclipped"),
    )
    mode_newton_relative = _scop_mode_newton_relative(mode)
    mode_tolerance = _scop_mode_tolerance(mode)
    if mode_newton_relative > mode_tolerance:
        if _certification_retry < 3:
            # Rungs 0->1 and 1->2 tighten the inner tolerance and re-fit from the
            # mode that just failed. That rescues a residual left by loose inner
            # convergence -- measured, 29 of 43 retries -- but it cannot move a
            # fit already converged tighter than the bar, which returns the same
            # mode bit-identically however hard the tolerance is squeezed.
            #
            # Rung 2->3 is therefore the cold one. It holds the tightest
            # tolerance the ladder reached rather than squeezing further, so it
            # differs from its predecessor in exactly one respect: the starting
            # point. The warm start it drops comes from a bootstrap fitted at
            # lambda=1e-4, a long way from these lambdas, which is the plausible
            # reason the mode landed off-stationary in the first place.
            cold_rung = _certification_retry == 2
            retry_tolerance = 10.0 ** (-10 - min(_certification_retry, 1))
            retry_context = replace(
                context,
                pirls_tol=min(context.pirls_tol, retry_tolerance),
            )
            # Rungs 0->1 and 1->2 are warm whatever the design's centring. They
            # once went cold whenever the raw-centring check failed (a 0/1
            # column carrying over half the IRLS weight fails it), which no
            # rationale required: the retry is an ordinary inner fit whose
            # solver chooses its centred system afresh at every iteration, and
            # its mode is certified afresh, so where it starts cannot change
            # what it must satisfy. Measured on the cleaned 678k-row book: 20
            # cold retries took 808 inner iterations; warm and Newton-polished
            # they took 20, with the same 57 outer iterations, deviance equal
            # to 1e-16 relative and EDF to 2e-14.
            warm_retry = not cold_rung
            retry_beta = result.beta.copy() if warm_retry else None
            retry_intercept = float(result.intercept) if warm_retry else None
            retry_states = scop_states if scop_states and warm_retry else None
            # Rung 0 -> 1 starts from the certificate's own Newton step; a
            # rung that already started there and still failed falls back to
            # the plain warm start, so the ladder keeps its original rescue.
            if warm_retry and scop_states and _certification_retry == 0:
                polished = _newton_polished_warm_start(mode, _scop_mode_newton_correction(mode))
                if polished is not None:
                    retry_beta, retry_intercept, retry_states = polished
            return _fit_scop_reml_mode(
                retry_context,
                lambdas,
                beta_init=retry_beta,
                intercept_init=retry_intercept,
                scop_state_init=retry_states,
                phase=phase,
                reml_iteration=reml_iteration,
                line_search_iteration=line_search_iteration,
                trial_alpha=trial_alpha,
                require_converged=require_converged,
                _certification_retry=_certification_retry + 1,
            )
        if require_converged:
            return None
        raise _scop_certification_failure(
            mode_newton_relative, mode_tolerance, mode_score.relative_max
        )
    if trace_run is not None and trace_run.enabled:
        if result.state_id is None:  # pragma: no cover - trace contract
            raise RuntimeError("traced SCOP REML evaluation is missing its coefficient state ID")
        phi = _reml_evaluation_phi(
            mode.evaluation,
            scale_known=getattr(context.distribution, "scale_known", True),
            fallback_likelihood_size=context.likelihood_size,
        )
        trace_run.emit_lazy(
            "evaluation",
            lambda: {
                "state_id": result.state_id,
                "evaluation_id": result.evaluation_id,
                "solver": "scop_efs_reml",
                "phase": phase,
                "outer_iteration": reml_iteration,
                "line_search_iteration": line_search_iteration,
                "trial_alpha": trial_alpha,
                "objective": mode.objective,
                "lambdas": mode.lambdas,
                "dispersion": phi,
                # Candidate rank/EDF work is deliberately omitted. Only the
                # public terminal is hydrated with authoritative SCOP EDF.
                "effective_df": None,
                "curvature_source": mode.curvature_source,
                "mode_score_relative": mode.mode_score.relative_max,
            },
            channel="reml",
            purpose=trace_purpose,
            authoritative=False,
        )
    return mode


def _certified_terminal_rank(
    matrix: NDArray,
    factor_factory: Callable[[], NDArray],
) -> RankDecomposition:
    """Apply the shared Gram-first policy to one terminal-only rank claim."""
    decomposition = decompose_gram_if_authoritative(matrix)
    if decomposition is None:
        decomposition = decompose_factor(factor_factory())
    return decomposition


def _hydrate_scop_terminal_rank_info(
    context: _SCOPREMLFitContext,
    mode: _SCOPREMLMode,
    *,
    fisher_weights: Callable[[], NDArray],
) -> None:
    """Install generic estimability metadata once on the retained SCOP mode."""
    result = mode.result
    if result.rank_info is not None:
        raise RuntimeError("SCOP REML candidate unexpectedly retained public rank metadata")

    data_rank = _certified_terminal_rank(
        mode.centered_xtwx,
        lambda: grouped_weighted_factor(
            context.dm,
            fisher_weights(),
            center=mode.fisher_mean_x,
        ),
    )
    augmented_rank = _certified_terminal_rank(
        mode.centered_xtwx + mode.penalty,
        lambda: grouped_augmented_factor(
            context.dm,
            fisher_weights(),
            mode.penalty,
            center=mode.fisher_mean_x,
        ),
    )
    coefficient_rank = _certified_terminal_rank(
        mode.xtwx + mode.penalty,
        lambda: grouped_augmented_factor(
            context.dm,
            fisher_weights(),
            mode.penalty,
        ),
    )

    selected_columns = np.arange(context.dm.p, dtype=int)
    selected_columns.setflags(write=False)
    mean_x = np.array(mode.fisher_mean_x, dtype=np.float64, copy=True)
    mean_x.setflags(write=False)
    feature_edf = np.zeros(context.dm.p, dtype=np.float64)
    feature_edf.setflags(write=False)
    result.rank_info = RankInfo(
        policy_version=SHARED_RANK_POLICY.version,
        coordinate_space="solver",
        selected_columns=selected_columns,
        selected_group_names=tuple(group.name for group in context.groups),
        sum_w=mode.fisher_sum_w,
        mean_x=mean_x,
        intercept_edf=0.0,
        data=data_rank,
        augmented=augmented_rank,
        coefficient=coefficient_rank,
        feature_edf=feature_edf,
        group_edf={group.name: 0.0 for group in context.groups},
        objective_loss=None,
    )


def _finalize_scop_reml_mode(
    context: _SCOPREMLFitContext,
    mode: _SCOPREMLMode,
) -> Any:
    """Hydrate exactly one retained coefficient mode for public inference."""
    result = mode.result
    if result.scop_inference is not None or result.scop_geometry is not None:
        raise RuntimeError("SCOP REML terminal mode was already hydrated")

    cached_weights: NDArray | None = None

    def terminal_fisher_weights() -> NDArray:
        nonlocal cached_weights
        if cached_weights is None:
            eta = stabilize_eta(
                context.dm.matvec(result.beta) + result.intercept + context.offset_arr,
                context.link,
            )
            mu = clip_mu(context.link.inverse(eta), context.distribution)
            cached_weights = fisher_working_weights(
                distribution=context.distribution,
                link=context.link,
                mu=mu,
                eta=eta,
                sample_weight=context.sample_weight,
            )
        return cached_weights

    _hydrate_scop_terminal_rank_info(
        context,
        mode,
        fisher_weights=terminal_fisher_weights,
    )
    result.phi = _reml_evaluation_phi(
        mode.evaluation,
        scale_known=getattr(context.distribution, "scale_known", True),
        fallback_likelihood_size=context.likelihood_size,
    )
    result.log_det_H = mode.log_det_h
    result.reml_hessian_rank = mode.hessian_rank
    fisher_xtw = mode.fisher_sum_w * mode.fisher_mean_x
    install_scop_postfit_inference(
        result,
        raw_fisher_gram=mode.xtwx,
        centered_fisher_gram=mode.centered_xtwx,
        fisher_xtw=fisher_xtw,
        fisher_mean_x=mode.fisher_mean_x,
        fisher_sum_w=mode.fisher_sum_w,
        latent_penalty=mode.penalty,
        scop_states=mode.scop_states,
        groups=context.groups,
        observed_geometry=mode.joint_geometry,
        dm=context.dm,
        fisher_weights=terminal_fisher_weights,
    )
    if not np.isfinite(result.phi) or not np.isfinite(result.effective_df):
        raise RuntimeError("SCOP REML terminal hydration left non-finite fit statistics")
    return result


def _backoff_scop_candidate_step(
    context: _SCOPREMLFitContext,
    origin: _SCOPREMLMode,
    proposed_lambdas: dict[str, float],
    *,
    reml_iteration: int,
) -> tuple[_SCOPREMLMode, dict[str, float], float] | None:
    """Retry a failed candidate at damped steps toward its certified origin.

    The candidate consumes the one EFS proposal that never went through the
    line search, so a certification failure there had no damping behind it
    and aborted the fit -- the same rejection the line search survives by
    trying the next alpha.  This applies the line search's trial formula,
    geometric interpolation in log-lambda at alpha = 0.5**attempt, between
    the mode the step was taken from and the proposal that failed, warm-
    starting each attempt from the origin.  The full step (alpha = 1.0) was
    the original candidate fit, so the attempts complete the same forward
    ladder the line search runs.

    Returns the first certified mode with its lambdas and the damping that
    produced it, or ``None`` when every damped attempt fails, in which case
    the caller keeps its raise.  A proposal-only name (absent from the
    origin) has no endpoint to interpolate from and is adopted undamped at
    its proposed value, keeping the caller's key set intact.
    """
    changed_names = [
        name
        for name, proposed in proposed_lambdas.items()
        if name in origin.lambdas and proposed != origin.lambdas[name]
    ]
    if not changed_names:
        return None

    log_directions: dict[str, float] = {}
    for name in changed_names:
        old = float(origin.lambdas[name])
        proposed = float(proposed_lambdas[name])
        if old <= 0.0 or proposed <= 0.0 or not np.isfinite(old + proposed):
            raise ValueError("SCOP EFS lambda trials must be positive and finite")
        log_directions[name] = float(np.log(proposed) - np.log(old))

    for attempt in range(1, _SCOP_EFS_MAX_BACKTRACK_ATTEMPTS):
        alpha = 0.5**attempt
        # Seeded from the proposal, not the origin: the adopted dict must
        # keep the loop's key set, so a name the origin never carried
        # survives at its proposed value instead of vanishing.
        trial_lambdas = proposed_lambdas.copy()
        for name in changed_names:
            log_trial = np.log(origin.lambdas[name]) + alpha * log_directions[name]
            trial_lambdas[name] = float(np.clip(np.exp(log_trial), 1.0e-6, 1.0e10))
        mode = _fit_scop_reml_mode(
            context,
            trial_lambdas,
            beta_init=origin.result.beta,
            intercept_init=float(origin.result.intercept),
            scop_state_init=origin.scop_states if origin.scop_states else None,
            phase="candidate",
            reml_iteration=reml_iteration,
            trial_alpha=alpha,
            require_converged=True,
        )
        if mode is not None:
            return mode, trial_lambdas, alpha
    return None


def _backtrack_scop_efs_candidate(
    context: _SCOPREMLFitContext,
    current: _SCOPREMLMode,
    proposed_lambdas: dict[str, float],
    *,
    reml_iteration: int,
    max_attempts: int = _SCOP_EFS_MAX_BACKTRACK_ATTEMPTS,
    reflect: bool = True,
) -> tuple[_SCOPREMLMode, bool]:
    """Fit and score repeatedly damped log-lambda trials.

    The returned boolean is false only when every attempted converged candidate
    in both the proposed and reflected directions is uphill (or every inner
    solve fails).  Reflection is a safeguarded fallback for EFS directions,
    whose expected-curvature update need not remain a descent direction for
    the exact LAML objective near a mode.  ``reflect=False`` tries the
    proposed direction only: a Newton step whose forward trials all fail is
    handed back to EFS rather than reflected.  In the failure case the exact
    current fitted mode is returned, so callers cannot accidentally publish an
    unevaluated lambda movement.

    Contract relied on by the candidate-site rescue guard: whenever no trial
    was objective-endorsed -- the failure case above, and the no-op early
    exit when the proposal changes nothing -- the returned mode is the
    *identical* ``current`` object, never a copy, so callers can detect
    "no acceptance gate saw a new state" by identity.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    changed_names = [
        name
        for name, proposed in proposed_lambdas.items()
        if name in current.lambdas and proposed != current.lambdas[name]
    ]
    if not changed_names:
        return current, True

    log_directions: dict[str, float] = {}
    for name in changed_names:
        old = float(current.lambdas[name])
        proposed = float(proposed_lambdas[name])
        if old <= 0.0 or proposed <= 0.0 or not np.isfinite(old + proposed):
            raise ValueError("SCOP EFS lambda trials must be positive and finite")
        log_directions[name] = float(np.log(proposed) - np.log(old))

    trial_number = 0
    for direction_sign in (1.0, -1.0) if reflect else (1.0,):
        direction_attempts = (
            max_attempts
            if direction_sign > 0.0
            else min(max_attempts, _SCOP_EFS_MAX_REFLECTED_ATTEMPTS)
        )
        for attempt in range(direction_attempts):
            alpha = direction_sign * 0.5**attempt
            trial_lambdas = current.lambdas.copy()
            for name in changed_names:
                log_trial = np.log(current.lambdas[name]) + alpha * log_directions[name]
                trial_lambdas[name] = float(np.clip(np.exp(log_trial), 1.0e-6, 1.0e10))

            trial_number += 1
            candidate = _fit_scop_reml_mode(
                context,
                trial_lambdas,
                beta_init=current.result.beta,
                intercept_init=float(current.result.intercept),
                scop_state_init=current.scop_states if current.scop_states else None,
                phase="line_search",
                reml_iteration=reml_iteration,
                line_search_iteration=trial_number,
                trial_alpha=alpha,
                require_converged=True,
            )
            if candidate is None:
                continue
            tolerance = _SCOP_LINE_SEARCH_RELATIVE_TOLERANCE * max(abs(current.objective), 1.0)
            candidate_is_acceptable = (
                candidate.objective <= current.objective + tolerance
                if direction_sign > 0.0
                else candidate.objective < current.objective
            )
            if np.isfinite(candidate.objective) and candidate_is_acceptable:
                return candidate, True

    return current, False


def _is_scop_component(pc: PenaltyComponent, scop_states: dict[int, dict]) -> dict | None:
    """Return SCOP state dict if pc corresponds to a SCOP group, else None."""
    state = scop_states.get(pc.group_index)
    if state is None:
        return None
    if state["group_name"] != pc.group_name or state["group_sl"] != pc.group_sl:
        return None
    return state


def scop_efs_lambda_update(
    pc: PenaltyComponent,
    beta: NDArray,
    H_joint_inv: NDArray,
    inv_phi: float,
    lam_old: float,
    scop_states: dict[int, dict],
) -> float:
    """Fellner-Schall fixed-point update for one penalty component.

    .. deprecated:: Use ``_joint_efs_lambda_step`` for the main EFS loop.
        This function uses the old fixed-point formula and is kept only for
        backward compatibility.
    """
    sl = pc.group_sl

    scop_st = _is_scop_component(pc, scop_states)
    if scop_st is not None:
        beta_g = scop_st["beta_eff"]
    else:
        beta_g = beta[sl]

    if np.linalg.norm(beta_g) < 1e-12:
        return lam_old

    quad = penalty_component_quadratic(pc, beta_g)
    trace_term = penalty_component_trace(pc, H_joint_inv[sl, sl])

    r_j = pc.rank
    denom = inv_phi * quad + trace_term

    if denom > 1e-12:
        lam_raw = r_j / denom
    else:
        return lam_old

    log_step = np.log(max(lam_raw, 1e-10)) - np.log(max(lam_old, 1e-10))
    log_step = np.clip(log_step, -5.0, 5.0)
    lam_new = lam_old * float(np.exp(log_step))

    return lam_new


def fit_fixed_scop_reml(
    dm: DesignMatrix,
    distribution,
    link,
    groups: list[GroupSlice],
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    lambdas: dict[str, float],
    *,
    weight_semantics: str,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
    reml_penalties: list[PenaltyComponent] | None = None,
    convergence: str = "deviance",
    _scop_joint: bool = True,
    debug_recorder=None,
) -> REMLResult:
    """Evaluate and publish one fixed-lambda SCOP coefficient mode."""
    (
        _gaussian_likelihood_size,
        saturated_log_weight,
        gamma_scale_data,
        tweedie_scale_data,
    ) = prepare_reml_scale_data(
        distribution,
        y,
        sample_weight,
        weight_semantics=weight_semantics,
    )
    # Every family needs the plain likelihood size here, not only Gaussian:
    # the SCOP path publishes a Pearson-style fallback dispersion for modes
    # whose exact profile is unavailable, and that denominator is the
    # contract's size whatever the family.  For Gaussian it is the same number
    # the scale profiler prepared.
    likelihood_size = dispersion_likelihood_size(
        sample_weight,
        weight_semantics=weight_semantics,
    )
    context = _SCOPREMLFitContext(
        dm=dm,
        distribution=distribution,
        link=link,
        groups=groups,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        reml_penalties=reml_penalties,
        convergence=convergence,
        scop_joint=_scop_joint,
        debug_recorder=debug_recorder,
        likelihood_size=likelihood_size,
        saturated_log_weight=saturated_log_weight,
        weight_semantics=weight_semantics,
        gamma_scale_data=gamma_scale_data,
        tweedie_scale_data=tweedie_scale_data,
    )
    mode = _fit_scop_reml_mode(
        context,
        lambdas,
        beta_init=None,
        intercept_init=None,
        scop_state_init=None,
        phase="fixed",
        reml_iteration=0,
        require_converged=True,
    )
    if mode is None:
        raise ObservedModeNotConvergedError(
            "fixed-lambda SCOP fit did not converge to a coefficient mode"
        )
    result = _finalize_scop_reml_mode(context, mode)
    step_norms = {
        state["group_name"]: float(state.get("last_step_norm", 0.0))
        for state in mode.scop_states.values()
    }
    fisher_fallbacks = sum(
        bool(state.get("last_fisher_fallback", False)) for state in mode.scop_states.values()
    )
    return REMLResult(
        lambdas=mode.lambdas.copy(),
        pirls_result=result,
        n_reml_iter=0,
        converged=bool(result.converged),
        lambda_history=[mode.lambdas.copy()],
        objective=float(mode.evaluation.value),
        reml_penalties=mode.penalty_components,
        scop_states=mode.scop_states if mode.scop_states else None,
        inner_iter_history=[int(result.n_iter)],
        objective_history=[float(mode.evaluation.value)],
        curvature_source=mode.curvature_source,
        tweedie_scale_data=tweedie_scale_data,
        termination_reason="fixed_lambdas",
        scop_step_norms=[step_norms] if step_norms else None,
        scop_fisher_fallbacks=int(fisher_fallbacks),
    )


def _aitken_step(
    state: dict, name: str, prev_dlsp: dict[str, float], dlsp: float, scaled_step: float
) -> float:
    """The EFS step, replaced by its Aitken limit when the linear phase is evidenced.

    ``state[("jumped", name)]`` marks the accepted step before this call as
    outside the linear phase (an Aitken jump, or a Newton step): it does not
    enter the history, which starts afresh.
    """
    history = state.setdefault(name, [])
    if state.get(("jumped", name)):
        history.clear()
        state[("jumped", name)] = False
    elif prev_dlsp.get(name, 0.0) != 0.0:
        history.append(float(prev_dlsp[name]))
    if len(history) < 4 or dlsp == 0.0 or abs(history[-1]) >= _SCOP_EFS_AITKEN_MAX_LAST_STEP:
        return scaled_step
    s4, s3, s2, s1 = history[-4:]
    if not (s4 * s3 > 0 and s3 * s2 > 0 and s2 * s1 > 0 and s1 * scaled_step > 0):
        return scaled_step
    ratios = (s3 / s4, s2 / s3, s1 / s2)
    if not all(_SCOP_EFS_AITKEN_MIN_RATIO < r < _SCOP_EFS_AITKEN_MAX_RATIO for r in ratios):
        return scaled_step
    if max(ratios) - min(ratios) > _SCOP_EFS_AITKEN_RATIO_AGREEMENT:
        return scaled_step
    jump = scaled_step / (1.0 - ratios[-1])
    state[("jumped", name)] = True
    state["jumps"] = state.get("jumps", 0) + 1
    return float(np.sign(jump) * min(abs(jump), _SCOP_EFS_AITKEN_MAX_JUMP))


def _joint_efs_lambda_step(
    all_pcs: list[PenaltyComponent],
    beta: NDArray,
    H_joint_inv: NDArray,
    phi: float,
    lambdas: dict[str, float],
    estimated_names: set[str],
    scop_states: dict[int, dict],
    alpha: dict[str, float],
    prev_dlsp: dict[str, float],
    flat_out: set[str] | None = None,
    aitken_state: dict | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Joint EFS lambda step using rEDF/pSp (Wood & Fasiolo 2017, scasm-style).

    Update on log scale::

        rEDF = rank - lambda * sEDF
        dlsp = log(phi) + log(rEDF) - log(pSp * lambda)
        log_lambda_new = log_lambda_old + alpha * dlsp

    with adaptive alpha (halve on sign-flip, grow on stable direction) and
    suppression detection.

    Parameters
    ----------
    all_pcs : list of PenaltyComponent
        All penalty components (SSP + SCOP).
    beta : (p,) coefficient vector (gamma for SCOP groups).
    H_joint_inv : (p, p) inverse of joint Hessian.
    phi : scale parameter (1.0 for known-scale families).
    lambdas : current lambda values keyed by component name.
    estimated_names : set of names to update.
    scop_states : SCOP converged state dict.
    alpha : per-component adaptive step size (mutated in place).
    prev_dlsp : previous step directions for sign-flip detection.
    flat_out : set, optional
        Receives the components either suppression hold covers at this
        iterate (``_scop_suppression_holds``): those at a flat end of the
        criterion.
    aitken_state : dict, optional
        Per-component accepted-step history for the safeguarded Aitken jump
        (``_SCOP_EFS_AITKEN_*``); ``None`` disables it.

    Returns
    -------
    lambdas_new : updated lambda dict.
    alpha : updated adaptive step sizes.
    dlsp_accepted : step directions (for sign-flip tracking; caller should
        update prev_dlsp from the POST-DAMPING accepted step, not this raw value).
    """
    lambdas_new = lambdas.copy()
    dlsp_out: dict[str, float] = {}
    prior_edf, _ = compute_logdet_s_derivatives(lambdas, all_pcs)

    for pc in all_pcs:
        if pc.name not in estimated_names:
            continue

        sl = pc.group_sl

        scop_st = _is_scop_component(pc, scop_states)
        if scop_st is not None:
            beta_g = scop_st["beta_eff"]
        else:
            beta_g = beta[sl]

        if np.linalg.norm(beta_g) < 1e-12:
            dlsp_out[pc.name] = 0.0
            continue

        # pSp and sEDF
        # Kind-aware reductions, not a materialized penalty: an identity block
        # would otherwise allocate eye(L) and run an O(L^3) matmul every outer
        # EFS iteration to produce what is beta @ beta and trace(H_inv), and a
        # repeated block would rebuild kron(I_L, omega_local) each time. These
        # are exactly the kinds this guard newly admits.
        pSp = penalty_component_quadratic(pc, beta_g)
        sEDF = penalty_component_trace(pc, H_joint_inv[sl, sl])

        # Residual EDF — keep raw for suppression check, floor for log.
        # The rank shortcut is exact only for an isolated penalty block.  For
        # overlapping components Wood--Fasiolo's generalized update requires
        # lambda_j tr(S_lambda^- S_j), i.e. the log|S|+ derivative.
        rEDF_raw = prior_edf[pc.name] - lambdas[pc.name] * sEDF
        rEDF_used = max(rEDF_raw, 1e-7)

        # Log-scale step: dlsp = log(phi) + log(rEDF) - log(pSp * lambda)
        pSp_lam = max(pSp * lambdas[pc.name], 1e-300)
        dlsp = np.log(max(phi, 1e-300)) + np.log(rEDF_used) - np.log(pSp_lam)

        # Suppression detection (scasm-style), both holds flatness tests in EDF
        increase_held, decrease_held = _scop_suppression_holds(
            float(prior_edf[pc.name]),
            float(lambdas[pc.name]) * sEDF,
            pSp * float(lambdas[pc.name]) / max(phi, 1e-300),
        )
        if increase_held and dlsp > 0:
            dlsp = 0.0
        if decrease_held and dlsp < 0:
            dlsp = 0.0
        if flat_out is not None and (increase_held or decrease_held):
            flat_out.add(pc.name)

        # Adaptive alpha damping
        if pc.name in prev_dlsp and prev_dlsp[pc.name] != 0.0:
            same_sign = dlsp * prev_dlsp[pc.name] > 0
            if not same_sign:
                alpha[pc.name] *= 0.5
            elif alpha.get(pc.name, 1.0) < 2.0:
                alpha[pc.name] = min(2.0, alpha.get(pc.name, 1.0) * 1.2)

        a_j = alpha.get(pc.name, 1.0)

        # Cap step magnitude
        scaled_step = a_j * dlsp
        if aitken_state is not None:
            scaled_step = _aitken_step(aitken_state, pc.name, prev_dlsp, dlsp, scaled_step)
        max_step = 4.0
        if abs(scaled_step) > max_step:
            scaled_step = max_step * np.sign(scaled_step)

        # Apply step
        log_lam_new = np.log(max(lambdas[pc.name], 1e-10)) + scaled_step
        lambdas_new[pc.name] = float(np.clip(np.exp(log_lam_new), 1e-6, 1e10))
        dlsp_out[pc.name] = dlsp

    return lambdas_new, alpha, dlsp_out


# Newton on log lambda for the SCOP outer loop.
#
# Fellner-Schall converges linearly near its fixed point and no faster than
# Newton (Wood & Fasiolo 2017, corollary of Theorem 3). This step solves for
# the same fixed point with Newton's method. Write V for the LAML criterion,
# q_j = beta' S_j beta in the latent coordinates (beta_eff for a SCOP block),
# t_j = tr(H^-1 S_j) with H the intercept-profiled latent Hessian, and
# r_j = d log|S|+ / d rho_j. The working gradient
#     g_j = lambda_j q_j / phi + lambda_j t_j - r_j        (= 2 dV/d rho_j at fixed H)
# is zero exactly where the EFS update leaves lambda_j unchanged. Like EFS, the
# PQL and performance-oriented iterations (Breslow & Clayton 1993; Gu 1992) and
# the discrete engine's cached-W Newton, it leaves out of the gradient the
# change of H through the coefficients (Wood & Fasiolo 2017, section 3), so the
# fixed point, and the answer, are the ones EFS has always reached. With
# d beta / d rho_k = -lambda_k H^-1 S_k beta (implicit differentiation at the
# latent mode; Wood, Pya & Saefken 2016, section 3.1.3) the Jacobian of g is
#     dg_j/d rho_k = delta_jk (lambda_j q_j / phi + lambda_j t_j)
#                    - 2 lambda_j lambda_k beta' S_j H^-1 S_k beta / phi
#                    - lambda_j lambda_k tr(H^-1 S_k H^-1 S_j)
#                    - d2 log|S|+ / d rho_j d rho_k
#                    + d(1/phi)/dD_p (lambda_j q_j)(lambda_k q_k)
#                    + lambda_j d t_j / d rho_k through H(beta),
# the fifth term only for an estimated scale profiled out of V (D_p is the
# penalized deviance, whose rho-derivative at the mode is lambda_k q_k). The
# first four are the terms ``reml_direct_hessian`` forms for an unconstrained
# fit, read here in latent coordinates. The last is what the constrained case
# adds: H moves with a SCOP block's own coefficients through the map
# gamma(beta_eff), and ``_scop_reparam_jacobian_correction`` forms that part
# with the working weights held fixed. Without it the BonusMalus diagonal on
# the cleaned freMTPL2 book read 3.05 against a finite-difference 5.25, and the
# step alternated in sign at a ratio of -0.72; with it every entry matched to
# 1e-3. The weights' own change is left out of the Jacobian (that check put it
# under 1e-3 here) and, as in EFS, out of the gradient. Exact LAML Newton
# (Wood, Pya & Saefken 2016, sections 3.1.3-3.1.4) would add both of these
# H(beta) terms to the gradient as well, moving the fixed point to the LAML
# optimum; that is a different estimator, not a faster route to this one. The
# line search on the exact LAML keeps every step honest, and where a Newton
# step is uphill on it the guard in ``optimize_scop_efs_reml`` restarts the
# search with EFS steps.
_SCOP_NEWTON_MAX_LOG_STEP = 4.0
# Forward trials (alpha = 1, 1/2, 1/4) before the guard restarts the search
# with EFS. A Newton direction from a positive-definite model is a descent
# direction of the working criterion, so a short backtrack finds the decrease
# when the LAML agrees with it; when none of three trials is accepted the two
# disagree, and more halving only spends inner fits.
_SCOP_NEWTON_MAX_BACKTRACK_ATTEMPTS = 3
# The EFS plateau's step cap (``max_change < 0.01`` in the plateau test), read
# against the full Newton step when the guard decides between a plateau stop
# and an EFS restart.
_SCOP_NEWTON_PLATEAU_MAX_STEP = 0.01


@dataclass(frozen=True)
class _SCOPNewtonSystem:
    """The working gradient and Jacobian of the SCOP outer step at one mode."""

    names: tuple[str, ...]
    log_lambdas: NDArray
    gradient: NDArray
    hessian: NDArray
    increase_held: NDArray
    decrease_held: NDArray


def _scop_newton_system(
    all_pcs: list[PenaltyComponent],
    beta: NDArray,
    H_joint_inv: NDArray,
    phi: float,
    lambdas: dict[str, float],
    estimated_names: set[str],
    scop_states: dict[int, dict],
    *,
    inverse_phi_derivative: float = 0.0,
) -> _SCOPNewtonSystem | None:
    """The fixed-curvature gradient and Jacobian on log lambda, or None.

    Reads the same reductions the EFS step reads (``penalty_component_quadratic``,
    ``penalty_component_trace`` and the log-determinant derivatives), so the
    two steps share their fixed point to the last bit of those terms. A
    component whose block is numerically zero is left out, as the EFS step
    leaves it in place. Returns None when no component remains or a term is
    not finite.
    """
    prior_edf, logdet_hessian = compute_logdet_s_derivatives(lambdas, all_pcs)
    width = int(H_joint_inv.shape[0])
    inverse_phi = 1.0 / max(float(phi), 1e-300)
    names: list[str] = []
    slices: list[slice] = []
    penalty_products: list[NDArray] = []
    penalty_beta: list[NDArray] = []
    weighted_quadratic: list[float] = []
    weighted_trace: list[float] = []
    gradient: list[float] = []
    increase_held: list[bool] = []
    decrease_held: list[bool] = []
    for pc in all_pcs:
        if pc.name not in estimated_names:
            continue
        group_slice = pc.group_sl
        scop_state = _is_scop_component(pc, scop_states)
        beta_group = np.asarray(
            scop_state["beta_eff"] if scop_state is not None else beta[group_slice],
            dtype=np.float64,
        )
        if np.linalg.norm(beta_group) < 1e-12:
            continue
        lam = float(lambdas[pc.name])
        quadratic = penalty_component_quadratic(pc, beta_group)
        trace = penalty_component_trace(pc, H_joint_inv[group_slice, group_slice])
        held_up, held_down = _scop_suppression_holds(
            float(prior_edf[pc.name]), lam * trace, lam * quadratic * inverse_phi
        )
        # H^-1 [:, block] S_j, the factor every cross term below reads; the
        # local penalty is the block's own q x q matrix, never a p x p one.
        columns = H_joint_inv[:, group_slice]
        product = (
            columns
            if pc.penalty_kind == "identity"
            else columns @ penalty_component_dense_matrix(pc)
        )
        full_penalty_beta = np.zeros(width, dtype=np.float64)
        full_penalty_beta[group_slice] = penalty_component_matvec(pc, beta_group)
        names.append(pc.name)
        slices.append(group_slice)
        penalty_products.append(lam * product)
        penalty_beta.append(lam * full_penalty_beta)
        weighted_quadratic.append(lam * quadratic * inverse_phi)
        weighted_trace.append(lam * trace)
        gradient.append(lam * quadratic * inverse_phi + lam * trace - float(prior_edf[pc.name]))
        increase_held.append(held_up)
        decrease_held.append(held_down)
    m = len(names)
    if m == 0:
        return None
    hessian = np.zeros((m, m), dtype=np.float64)
    weighted_penalty_quadratic = np.asarray(weighted_quadratic, dtype=np.float64) / inverse_phi
    for j in range(m):
        inverse_penalty_beta_j = H_joint_inv @ penalty_beta[j]
        for k in range(j, m):
            cross_beta = 2.0 * inverse_phi * float(penalty_beta[k] @ inverse_penalty_beta_j)
            # tr(H^-1 lambda_k S_k H^-1 lambda_j S_j) from the two column blocks.
            cross_trace = float(
                np.sum(penalty_products[k][slices[j], :] * penalty_products[j][slices[k], :].T)
            )
            logdet_term = float(
                logdet_hessian.get(
                    (names[j], names[k]), logdet_hessian.get((names[k], names[j]), 0.0)
                )
            )
            value = (
                -cross_beta
                - cross_trace
                - logdet_term
                + inverse_phi_derivative
                * weighted_penalty_quadratic[j]
                * weighted_penalty_quadratic[k]
            )
            if j == k:
                value += weighted_quadratic[j] + weighted_trace[j]
            hessian[j, k] = hessian[k, j] = value
    g = np.asarray(gradient, dtype=np.float64)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(hessian))):
        return None
    return _SCOPNewtonSystem(
        names=tuple(names),
        log_lambdas=np.log(np.array([float(lambdas[name]) for name in names])),
        gradient=g,
        hessian=hessian,
        increase_held=np.asarray(increase_held, dtype=bool),
        decrease_held=np.asarray(decrease_held, dtype=bool),
    )


def _scop_reparam_jacobian_correction(
    mode: _SCOPREMLMode, system: _SCOPNewtonSystem
) -> NDArray | None:
    """The working Jacobian's SCOP reparameterisation terms, or None.

    The part of ``d t_j / d rho_k = -tr(H^-1 (dH/d rho_k) H^-1 S_j)`` that
    comes from H moving with a SCOP block's own coefficients through the map
    gamma(beta_eff), with the working weights held fixed as everywhere in the
    step (Pya & Wood 2015 differentiate the same map; see the section comment
    for the measured need). On a positivity coordinate the map and its first
    three derivatives are all ``exp(beta_eff)``. Write K for the latent data
    curvature, ``c = J X'W1``, e for the positivity-coordinate indicator and
    ``(v0, v) = d(beta_0, beta) / d rho_k``. At the mode the latent penalty
    gradient ``S beta`` equals the map's derivative times the score, so
    ``H = K - diag(e S beta) + S`` and, along v,
    ``dK = E K + K E`` with ``E = diag(e v)``, the diagonal term moves by
    ``e (K v + c v0 - v S beta)`` and ``dc = E c``; the intercept-profiled H
    loses ``(dc c' + c dc') / sum(W)``. Returned unsymmetrised, as the j, k
    entries of ``lambda_j d t_j / d rho_k``. None when no block carries a
    positivity coordinate, a block's Newton solve or the joint geometry fell
    back to Fisher curvature (the curvature then lacks the map's term), or
    the map is not the exp map this assumes.
    """
    if mode.curvature_source != "observed":
        return None
    geometry = mode.joint_geometry
    hessian_inverse = mode.hessian_inverse
    width = int(hessian_inverse.shape[0])
    positivity = np.zeros(width, dtype=np.float64)
    latent_beta = np.asarray(mode.result.beta, dtype=np.float64).copy()
    for state in mode.scop_states.values():
        if bool(state.get("last_fisher_fallback", False)):
            return None
        group_slice = state["group_sl"]
        beta_eff = np.asarray(state["beta_eff"], dtype=np.float64)
        latent_beta[group_slice] = beta_eff
        jacobian = np.asarray(state["reparam"].jacobian_diagonal(beta_eff), dtype=np.float64)
        second = np.asarray(state["reparam"].second_derivative_diagonal(beta_eff), dtype=np.float64)
        ratio = np.divide(second, jacobian, out=np.zeros_like(second), where=jacobian != 0.0)
        if not np.all((ratio == 0.0) | (ratio == 1.0)):
            return None
        positivity[group_slice] = ratio
    if not np.any(positivity):
        return None
    cross = np.asarray(geometry.transformed_intercept_cross, dtype=np.float64)
    sum_w = float(geometry.sum_w)
    penalty_gradient = mode.penalty @ latent_beta
    curvature = (
        np.asarray(geometry.centered_hessian, dtype=np.float64)
        + np.outer(cross, cross) / sum_w
        - mode.penalty
        + np.diag(positivity * penalty_gradient)
    )
    components = [
        next(pc for pc in mode.penalty_components if pc.name == name) for name in system.names
    ]
    # Columns are the m directions (v0_k, v_k), from lambda_k S_k beta.
    penalty_beta = np.zeros((width, len(components)), dtype=np.float64)
    for k, pc in enumerate(components):
        penalty_beta[pc.group_sl, k] = float(mode.lambdas[pc.name]) * penalty_component_matvec(
            pc, latent_beta[pc.group_sl]
        )
    directions = -(hessian_inverse @ penalty_beta)
    intercept_directions = -(cross @ directions) / sum_w
    scaled = positivity[:, None] * directions
    diagonal_moves = positivity[:, None] * (
        curvature @ directions
        + np.outer(cross, intercept_directions)
        - directions * penalty_gradient[:, None]
    )
    cross_changes = scaled * cross[:, None]
    # With M_j = H^-1 S_j H^-1, tr(H^-1 dH_k H^-1 S_j) = <dH_k, M_j>, and each
    # piece of dH_k is a diagonal scaling of K, a diagonal, or a rank-two term:
    #     <dH_k, M_j> = 2 s_k . diag(K M_j) + d_k . diag(M_j) - 2 (dc_k . M_j c) / sum(W),
    # O(p^2 q_j) per component instead of a p^3 product per direction.
    correction = np.empty((len(components), len(components)), dtype=np.float64)
    for j, pc in enumerate(components):
        columns = hessian_inverse[:, pc.group_sl]
        local = (
            columns
            if pc.penalty_kind == "identity"
            else columns @ penalty_component_dense_matrix(pc)
        )
        sandwich = local @ columns.T
        correction[j, :] = -float(mode.lambdas[pc.name]) * (
            2.0 * (np.sum(curvature * sandwich, axis=1) @ scaled)
            + np.diag(sandwich) @ diagonal_moves
            - 2.0 * ((sandwich @ cross) @ cross_changes) / sum_w
        )
    if not np.all(np.isfinite(correction)):
        return None
    return correction


def _scop_newton_step(system: _SCOPNewtonSystem, jacobian: NDArray | None = None) -> NDArray | None:
    """The safeguarded Newton step on log lambda, zero on held components.

    A component is held, as the EFS step holds it, when its own descent
    direction (the sign of ``-g_j``) points into a suppression hold; the step
    is then solved on the rest, and any component whose solved step still
    points into its hold is held and the step solved again. ``jacobian``, when
    given, replaces the system's fixed-curvature one (the caller passes it with
    the SCOP reparameterisation terms added and symmetrised). The Jacobian is
    made positive definite by its absolute eigenvalues with a relative floor
    (Wood, Pya & Saefken 2016, outer step 4d), so the step is a descent
    direction of the working criterion, and the longest coordinate is capped at
    ``_SCOP_NEWTON_MAX_LOG_STEP``. Returns None when the step is not finite.
    """
    g = system.gradient
    matrix = system.hessian if jacobian is None else jacobian
    m = g.size
    free = ~((system.increase_held & (g < 0.0)) | (system.decrease_held & (g > 0.0)))
    step = np.zeros(m, dtype=np.float64)
    for _ in range(m):
        index = np.flatnonzero(free)
        step = np.zeros(m, dtype=np.float64)
        if index.size == 0:
            break
        values, vectors = np.linalg.eigh(matrix[np.ix_(index, index)])
        floor = np.finfo(np.float64).eps ** 0.7 * max(float(np.max(np.abs(values))), 1e-300)
        values = np.maximum(np.abs(values), floor)
        step[index] = -(vectors @ ((vectors.T @ g[index]) / values))
        blocked = free & (
            (system.increase_held & (step > 0.0)) | (system.decrease_held & (step < 0.0))
        )
        if not np.any(blocked):
            break
        free &= ~blocked
    step[~free] = 0.0
    largest = float(np.max(np.abs(step), initial=0.0))
    if not np.isfinite(largest):
        return None
    if largest > _SCOP_NEWTON_MAX_LOG_STEP:
        step *= _SCOP_NEWTON_MAX_LOG_STEP / largest
    return step


def optimize_scop_efs_reml(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    lambdas: dict[str, float],
    estimated_names: set[str],
    *,
    weight_semantics: str,
    max_reml_iter: int = 20,
    reml_tol: float = 1e-6,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
    verbose: bool = False,
    reml_penalties: list[PenaltyComponent] | None = None,
    convergence: str = "deviance",
    _scop_joint: bool = True,
    debug_recorder=None,
    warm_lambdas: Mapping[str, float] | None = None,
    _aitken: bool = True,
    _outer_step: str = "newton",
) -> REMLResult:
    """SCOP-aware REML optimizer for monotone splines.

    Solves for the Fellner-Schall fixed point (Wood & Fasiolo 2017) using
    ``fit_irls_direct`` with SCOP Newton inner solver. Each outer iteration:

    1. Fit via ``fit_irls_direct(return_xtwx=True, return_scop_state=True)``
    2. Build the joint Hessian with SCOP Newton blocks replacing linear blocks
    3. Propose a Newton step on log lambda toward the Fellner-Schall fixed
       point (``_scop_newton_system``), or an EFS update using SCOP-aware quad
       and trace terms, replaced by their safeguarded Aitken limit once a
       component's steps contract at a stable ratio (``_aitken``)
    4. Step-damp via REML objective comparison
    5. Check convergence on max abs log-lambda change

    The Newton step is the default (``_outer_step="newton"``). The fit hands
    itself to EFS for the rest of the run, and records why in
    ``REMLResult.scop_newton_fallback``, when the Newton step is refused (an
    estimated-scale family without a profiled scale, the multi-SCOP discrete
    cleanup, or a non-finite system). When a Newton step fails its guard (no
    forward trial accepted), a step under the plateau's 0.01 cap whose
    predicted decrease the objective cannot resolve ends the run as an
    objective plateau; at the first iteration EFS takes over in place
    (``"line_search_first_iteration"``); at the iteration cap the run stops
    there, unconverged (``"line_search_at_cap"``); and otherwise the search
    restarts from the bootstrap with EFS steps (``"line_search"``), within
    the iterations left, and returns what that EFS search returns within
    them, with the Newton iterations prepended to its histories. At an
    iterate where a SCOP block's inner solve, or the joint geometry, fell
    back to Fisher curvature the Newton Jacobian lacks its
    reparameterisation terms, so that iteration takes an EFS step instead
    (``"efs_fisher"`` in ``scop_outer_steps``).
    ``_outer_step="efs"`` runs EFS throughout.

    Parameters
    ----------
    dm : DesignMatrix
        Design matrix (discretized for SCOP).
    distribution : Distribution
        GLM family.
    link : Link
        Link function.
    groups : list of GroupSlice
        Group definitions for each feature.
    y : ndarray
        Response vector.
    sample_weight : ndarray
        Observation weights, read under the ``weight_semantics`` supplied
        for Tweedie.
    offset_arr : ndarray
        Offset vector.
    lambdas : dict
        Initial smoothing parameters keyed by group name.
    estimated_names : set of str
        Names of lambda components to estimate (others held fixed).
    max_reml_iter : int
        Maximum outer EFS iterations.
    reml_tol : float
        Convergence tolerance on max abs log-lambda change.
    pirls_tol : float
        Convergence tolerance for inner IRLS solver.
    max_pirls_iter : int
        Maximum inner IRLS iterations.
    verbose : bool
        Print iteration progress.
    reml_penalties : list of PenaltyComponent, optional
        Pre-built penalty components for non-SCOP terms.
    convergence : str
        PIRLS convergence criterion: 'deviance' or 'coefficients'.
    warm_lambdas : mapping, optional
        A warm start (a previous fit's estimates, keyed by component name).
        The bootstrap fit is taken at these values and the first EFS update
        starts from them; components it does not name bootstrap cold at 1e-4.

    Returns
    -------
    REMLResult
        Result with estimated lambdas, final PIRLS result, convergence info.
    """
    scale_known = getattr(distribution, "scale_known", True)
    (
        _gaussian_likelihood_size,
        saturated_log_weight,
        gamma_scale_data,
        tweedie_scale_data,
    ) = prepare_reml_scale_data(
        distribution,
        y,
        sample_weight,
        weight_semantics=weight_semantics,
    )
    # Every family needs the plain likelihood size here, not only Gaussian:
    # the SCOP path publishes a Pearson-style fallback dispersion for modes
    # whose exact profile is unavailable, and that denominator is the
    # contract's size whatever the family.  For Gaussian it is the same number
    # the scale profiler prepared.
    likelihood_size = dispersion_likelihood_size(
        sample_weight,
        weight_semantics=weight_semantics,
    )
    fit_context = _SCOPREMLFitContext(
        dm=dm,
        distribution=distribution,
        link=link,
        groups=groups,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        reml_penalties=reml_penalties,
        convergence=convergence,
        scop_joint=_scop_joint,
        debug_recorder=debug_recorder,
        likelihood_size=likelihood_size,
        saturated_log_weight=saturated_log_weight,
        weight_semantics=weight_semantics,
        gamma_scale_data=gamma_scale_data,
        tweedie_scale_data=tweedie_scale_data,
    )
    # The caller's starting values, kept for a guard restart (see the docstring).
    start_lambdas = dict(lambdas)
    lambdas = lambdas.copy()

    # -- Bootstrap: one IRLS with minimal penalty -> one EFS step --
    # Fixed-policy lambdas keep their value; only estimated components get 1e-4,
    # unless a warm start names them: then the bootstrap is fitted at the warm
    # value and the EFS step below leaves it there, so the first outer
    # iteration reuses this very mode instead of fitting it again.
    warm_names = {
        name for name in estimated_names if warm_lambdas is not None and name in warm_lambdas
    }
    boot_lambdas = {
        name: (
            float(np.clip(warm_lambdas[name], 1e-6, 1e10))
            if name in warm_names
            else (1e-4 if name in estimated_names else val)
        )
        for name, val in lambdas.items()
    }
    boot_mode = _fit_scop_reml_mode(
        fit_context,
        boot_lambdas,
        beta_init=None,
        intercept_init=None,
        scop_state_init=None,
        phase="bootstrap",
        reml_iteration=0,
        require_converged=True,
    )
    if boot_mode is None:
        raise ObservedModeNotConvergedError(
            "SCOP REML bootstrap did not converge to a coefficient mode"
        )
    boot_result = boot_mode.result
    boot_scop_states = boot_mode.scop_states
    all_pcs = boot_mode.penalty_components
    H_joint_inv_boot = boot_mode.hessian_inverse
    boot_evaluation = boot_mode.evaluation
    boot_phi = _reml_evaluation_phi(
        boot_evaluation,
        scale_known=scale_known,
        fallback_likelihood_size=fit_context.likelihood_size,
    )

    # One EFS step on bootstrap beta — uses rEDF/pSp formula for ALL terms
    # (including SCOP). This gives SCOP lambdas their first meaningful move.
    boot_alpha = {name: 1.0 for name in estimated_names}
    boot_lambdas_new, _, _ = _joint_efs_lambda_step(
        all_pcs,
        boot_result.beta,
        H_joint_inv_boot,
        boot_phi,
        {pc.name: boot_lambdas.get(pc.name, 1e-4) for pc in all_pcs},
        estimated_names,
        boot_scop_states,
        boot_alpha,
        {},
    )
    for name in estimated_names:
        if name in warm_names:
            lambdas[name] = boot_lambdas[name]
        elif name in boot_lambdas_new:
            lambdas[name] = boot_lambdas_new[name]

    if verbose:
        lam_str = ", ".join(f"{pc.name}={lambdas[pc.name]:.4g}" for pc in all_pcs)
        print(f"  SCOP REML bootstrap: lambdas=[{lam_str}]")

    # -- Main EFS loop --
    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    converged = False
    termination_reason = "max_reml_iter"
    n_reml_iter = 0
    warm_beta: NDArray | None = boot_result.beta.copy()
    warm_intercept: float = float(boot_result.intercept)
    warm_scop_states: dict[int, dict] | None = boot_scop_states if boot_scop_states else None
    step_origin: _SCOPREMLMode = boot_mode
    # A complete warm start leaves the iterate at the bootstrap's own lambdas:
    # that coherent, certified mode is the first iteration's current mode.
    retained_mode: _SCOPREMLMode | None = boot_mode if boot_mode.lambdas == lambdas else None
    current_mode: _SCOPREMLMode | None = None

    # Convergence diagnostics
    flat_names: set[str] = set()
    inner_iter_history: list[int] = []
    objective_history: list[float] = []
    scop_step_norms_history: list[dict[str, float]] = []
    total_fisher_fallbacks = 0
    managed_cleanup_active_history: list[list[str]] = []
    managed_cleanup_frozen_history: list[list[str]] = []
    managed_cleanup_freeze_iter: int | None = None

    # Adaptive EFS step state (per-component)
    efs_alpha: dict[str, float] = {name: 1.0 for name in estimated_names}
    efs_prev_dlsp: dict[str, float] = {}
    aitken_state: dict | None = {} if _aitken else None

    scop_term_count = sum(1 for group in groups if group.monotone_engine == "scop")
    managed_cleanup_names = _multi_scop_discrete_cleanup_names(
        estimated_names=estimated_names,
        scop_states=boot_scop_states,
        scop_term_count=scop_term_count,
    )
    managed_cleanup_active = bool(managed_cleanup_names)
    active_names: set[str] = set(estimated_names)
    frozen_names: set[str] = set()
    stable_counts: dict[str, int] = {name: 0 for name in managed_cleanup_names}
    # Max accepted log-lambda step of the previous iteration plus a count of
    # consecutive non-contracting observations: the plateau exit uses them
    # to tell "still buying precision" from "stalled at the noise floor".
    accepted_change_window: list[float] = []
    # No observed contraction yet reads as r >= 1 (no geometric claim);
    # irrelevant until the window can fill.
    step_contraction_ratio = float("inf")

    # The outer step: Newton toward the Fellner-Schall fixed point, handed to
    # EFS for the rest of the run when it is refused or fails its guard. The
    # reason and the iteration are published on the result.
    if _outer_step not in ("newton", "efs"):
        raise ValueError(f"_outer_step must be 'newton' or 'efs', got {_outer_step!r}")
    newton_fallback: str | None = None
    newton_fallback_iter: int | None = None
    if _outer_step == "efs":
        newton_fallback, newton_fallback_iter = "requested", 0
    elif managed_cleanup_active:
        # The multi-SCOP discrete cleanup freezes floor-pinned components on
        # counts of accepted EFS updates; it keeps the step it was built on.
        newton_fallback, newton_fallback_iter = "multi_scop_cleanup", 0
    elif not scale_known and boot_evaluation.profiled_scale is None:
        # A custom estimated-scale family's Gaussian-shaped scale term has no
        # profiled d(1/phi)/dD_p for the Newton Jacobian.
        newton_fallback, newton_fallback_iter = "scale_profile", 0
    newton_live = newton_fallback is None
    outer_steps: list[str] = []
    restart_with_efs = False

    for reml_iter in range(max_reml_iter):
        n_reml_iter = reml_iter + 1

        # Step 1: Fit the current lambda mode, unless the preceding accepted
        # line-search trial already produced this exact coherent state.
        rescue_alpha: float | None = None
        if retained_mode is None:
            current_mode = _fit_scop_reml_mode(
                fit_context,
                lambdas,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                scop_state_init=warm_scop_states,
                phase="candidate",
                reml_iteration=n_reml_iter,
                require_converged=True,
            )
            if current_mode is None:
                # The one EFS proposal with no line search behind it; back the
                # step off toward the certified mode it was taken from.  The
                # bootstrap and fixed-lambda sites have no such mode and stay
                # fatal.
                rescue = _backoff_scop_candidate_step(
                    fit_context,
                    step_origin,
                    lambdas,
                    reml_iteration=n_reml_iter,
                )
                if rescue is None:
                    raise ObservedModeNotConvergedError(
                        "SCOP REML candidate did not converge to a coefficient mode"
                    )
                current_mode, lambdas, rescue_alpha = rescue
                # The failed proposal was never fitted; the history entry for
                # this step becomes the damped vector actually adopted, so
                # consumers of lambda_history only ever see fitted vectors.
                lambda_history[-1] = lambdas.copy()
                if verbose:
                    print(f"  SCOP REML candidate backoff: rescued at alpha={rescue_alpha:.4g}")
        else:
            current_mode = retained_mode
            retained_mode = None

        result = current_mode.result
        scop_states = current_mode.scop_states
        beta = result.beta
        inner_iter_history.append(result.n_iter)

        # Collect SCOP diagnostics from this inner fit
        step_norms_this_iter: dict[str, float] = {}
        for gi, st in scop_states.items():
            step_norms_this_iter[st["group_name"]] = st.get("last_step_norm", 0.0)
            if st.get("last_fisher_fallback", False):
                total_fisher_fallbacks += 1
        scop_step_norms_history.append(step_norms_this_iter)

        # Steps 2--5: Reuse the penalty, latent Hessian, and LAML evaluation
        # assembled from the same fitted coefficient mode.
        H_joint_inv = current_mode.hessian_inverse
        all_pcs = current_mode.penalty_components
        current_evaluation = current_mode.evaluation
        phi = _reml_evaluation_phi(
            current_evaluation,
            scale_known=scale_known,
            fallback_likelihood_size=fit_context.likelihood_size,
        )

        obj_curr = current_mode.objective
        objective_history.append(float(obj_curr))
        flat_names = set()
        retained_mode = None
        candidate_accepted = False
        step_kind = "efs"

        # Step 6a: Newton step on log lambda over active_names, line-searched
        # forward on the exact LAML (``_scop_newton_system``). Every candidate's
        # coefficients, SCOP blocks, penalty, and LAML geometry share the same
        # lambda state; a failed search retains the exact current mode.
        # A block whose inner solve fell back to Fisher curvature retains a
        # curvature without the map's term, and so does a joint geometry that
        # fell back to Fisher curvature itself (an indefinite observed Hessian,
        # or retained blocks that would not decompose): its H is J F J + S. The
        # reparameterisation correction reads H as K - diag(e S beta) + S, so
        # it cannot be formed on either, and without it the Jacobian's SCOP
        # diagonal measured 3.05 against a finite-difference 5.25: this
        # iteration takes the EFS step instead and Newton resumes at the next.
        fisher_iterate = newton_live and (
            current_mode.curvature_source != "observed"
            or any(bool(state.get("last_fisher_fallback", False)) for state in scop_states.values())
        )
        if newton_live and not fisher_iterate:
            inverse_phi_derivative: float | None = 0.0
            if not scale_known:
                profiled_scale = current_evaluation.profiled_scale
                try:
                    inverse_phi_derivative = (
                        None
                        if profiled_scale is None
                        else float(profiled_scale.d_inverse_phi_d_penalized_deviance)
                    )
                except FloatingPointError:
                    inverse_phi_derivative = None
            system = (
                None
                if inverse_phi_derivative is None or not np.isfinite(inverse_phi_derivative)
                else _scop_newton_system(
                    all_pcs,
                    beta,
                    H_joint_inv,
                    phi,
                    lambdas,
                    active_names,
                    scop_states,
                    inverse_phi_derivative=inverse_phi_derivative,
                )
            )
            # The reparameterisation terms enter symmetrised: the exact LAML
            # Hessian is symmetric, and what the fixed weights leave of the
            # asymmetry measured 2e-2 against diagonals of 1 to 7.
            correction = (
                None if system is None else _scop_reparam_jacobian_correction(current_mode, system)
            )
            newton_step = (
                None
                if system is None
                else _scop_newton_step(
                    system,
                    None
                    if correction is None
                    else system.hessian + 0.5 * (correction + correction.T),
                )
            )
            if system is None or newton_step is None:
                newton_live = False
                newton_fallback = (
                    "scale_profile"
                    if inverse_phi_derivative is None or not np.isfinite(inverse_phi_derivative)
                    else "newton_system"
                )
                newton_fallback_iter = n_reml_iter
            else:
                step_kind = "newton"
                flat_names = {
                    name
                    for name, held_up, held_down in zip(
                        system.names, system.increase_held, system.decrease_held, strict=True
                    )
                    if held_up or held_down
                }
                if rescue_alpha is None and float(np.max(np.abs(newton_step))) < reml_tol:
                    # The full Newton step is the distance to the fixed point
                    # of the local model, so a step under the tolerance
                    # certifies this mode as the EFS update's no-op would.
                    # A rescued mode still goes through the objective gate.
                    retained_mode, candidate_accepted = current_mode, True
                else:
                    proposal = lambdas.copy()
                    for name, log_lambda, delta in zip(
                        system.names, system.log_lambdas, newton_step, strict=True
                    ):
                        if delta != 0.0:
                            proposal[name] = float(np.clip(np.exp(log_lambda + delta), 1e-6, 1e10))
                    retained_mode, candidate_accepted = _backtrack_scop_efs_candidate(
                        fit_context,
                        current_mode,
                        proposal,
                        reml_iteration=n_reml_iter,
                        max_attempts=_SCOP_NEWTON_MAX_BACKTRACK_ATTEMPTS,
                        reflect=False,
                    )
                    if not candidate_accepted:
                        # The guard: no forward trial was accepted, so the
                        # exact LAML disagrees with the working model here
                        # (where the LAML optimum and the EFS fixed point
                        # separate, the step back to the fixed point can be
                        # uphill). Three cases, in this order.
                        #
                        # The step is inside the plateau: the full Newton
                        # step bounds the distance to the fixed point of the
                        # local model, as the EFS plateau's remaining-
                        # movement bound does, and here it is under the
                        # plateau's 0.01 step cap, while the decrease the
                        # model predicts for it, |g' delta| / 4 (g = 2 dV/d
                        # rho), is under the tolerance the line search
                        # accepts a trial within, so no objective gate can
                        # resolve it. The mode is published as an objective
                        # plateau, as EFS publishes the same state. A rescued
                        # mode (the first iteration's backed-off candidate)
                        # never is: no gate has endorsed it.
                        #
                        # No Newton step has been accepted yet (the first
                        # iteration): this mode, rescued or not, is the one
                        # the EFS search starts from, so EFS takes over here
                        # with the fresh state it would have there, and a
                        # rescued mode keeps its endorsement check below.
                        #
                        # No iteration is left (the cap): nothing can restart,
                        # so the run stops at this mode, unconverged on
                        # ``max_reml_iter``, and the fallback records that the
                        # last Newton step failed its line search there.
                        #
                        # Otherwise the search restarts from the bootstrap
                        # with EFS steps (below the loop). Continuing EFS
                        # from this mode was measured to fail: a Newton step
                        # that overshot the fixed point can land below it on
                        # the LAML, where EFS has no descent path back. Its
                        # forward trials were then rejected and its reflected
                        # ones accepted, and on a flat monotone truth it
                        # stalled unconverged after 28 iterations and 194
                        # inner fits, against 12 and 14 for EFS from the
                        # start.
                        retained_mode = None
                        newton_fallback, newton_fallback_iter = "line_search", n_reml_iter
                        predicted = 0.25 * abs(float(system.gradient @ newton_step))
                        if (
                            rescue_alpha is None
                            and float(np.max(np.abs(newton_step))) < _SCOP_NEWTON_PLATEAU_MAX_STEP
                            and predicted
                            <= _SCOP_LINE_SEARCH_RELATIVE_TOLERANCE
                            * max(abs(current_mode.objective), 1.0)
                        ):
                            newton_fallback = newton_fallback_iter = None
                            outer_steps.append("newton")
                            lambda_history.append(lambdas.copy())
                            converged = True
                            termination_reason = "objective_plateau"
                            break
                        elif n_reml_iter == 1:
                            newton_live = False
                            newton_fallback = "line_search_first_iteration"
                            flat_names = set()
                        else:
                            outer_steps.append("newton")
                            lambda_history.append(lambdas.copy())
                            restart_with_efs = n_reml_iter < max_reml_iter
                            if not restart_with_efs:
                                newton_fallback = "line_search_at_cap"
                            break

        # Step 6b: Joint EFS lambda update (rEDF/pSp, scasm-style), with its
        # own bounded line search (Step 7). Only update components in
        # active_names (frozen ones are skipped).
        if retained_mode is None:
            step_kind = "efs_fisher" if fisher_iterate else "efs"
            lambdas_new, efs_alpha, raw_dlsp = _joint_efs_lambda_step(
                all_pcs,
                beta,
                H_joint_inv,
                phi,
                lambdas,
                active_names,
                scop_states,
                efs_alpha,
                efs_prev_dlsp,
                flat_names,
                aitken_state,
            )
            retained_mode, candidate_accepted = _backtrack_scop_efs_candidate(
                fit_context,
                current_mode,
                lambdas_new,
                reml_iteration=n_reml_iter,
            )
        outer_steps.append(step_kind)
        lambdas_new = retained_mode.lambdas.copy()
        obj_after = retained_mode.objective
        # The line search returns the identical current mode in exactly the
        # two no-endorsement cases (every trial rejected, or a no-op proposal
        # accepted without fitting anything) -- a contract its docstring pins.
        # Computed here, not at the guard below, so the level-2 payload row
        # records the same verdict the guard acts on; retained_mode must not
        # be reassigned between the two points.
        rescue_endorsed: bool | None = None
        if rescue_alpha is not None:
            rescue_endorsed = retained_mode is not current_mode

        # Update prev_dlsp from ACCEPTED (post-damping) step
        for name in estimated_names:
            if name in lambdas_new and name in lambdas:
                accepted_step = np.log(max(lambdas_new[name], 1e-10)) - np.log(
                    max(lambdas[name], 1e-10)
                )
                efs_prev_dlsp[name] = accepted_step
        # Aitken extrapolates a run of EFS steps contracting at one ratio, and
        # a Newton step is no part of that run: the next EFS step (an
        # ``"efs_fisher"`` iterate, or the hand-off to EFS) starts the history
        # afresh, as after an Aitken jump (``_aitken_step``).
        if step_kind == "newton" and aitken_state is not None:
            for name in estimated_names:
                aitken_state[("jumped", name)] = True

        # Step 7b: Multi-SCOP discrete cleanup — freeze floor-pinned components
        # after they have been stable for several accepted lambda updates.
        if managed_cleanup_active and candidate_accepted:
            frozen_names_before = set(frozen_names)
            managed_active_names = managed_cleanup_names - frozen_names
            stable_counts = _update_multi_scop_discrete_stability_counts(
                lambdas_old=lambdas,
                lambdas_new=lambdas_new,
                active_names=managed_active_names,
                stable_counts=stable_counts,
            )
            managed_active_names, frozen_names = _freeze_multi_scop_discrete_lambdas(
                active_names=managed_active_names,
                frozen_names=frozen_names,
                lambdas_new=lambdas_new,
                stable_counts=stable_counts,
            )
            frozen_names &= managed_cleanup_names
            active_names = set(estimated_names) - frozen_names
            # Record the first 1-based outer iteration where the frozen set
            # changes relative to the previous accepted iteration.
            if managed_cleanup_freeze_iter is None and frozen_names != frozen_names_before:
                managed_cleanup_freeze_iter = n_reml_iter
            # Histories store accepted post-update snapshots for this outer step.
            managed_cleanup_active_history.append(sorted(managed_cleanup_names - frozen_names))
            managed_cleanup_frozen_history.append(sorted(frozen_names))
        elif managed_cleanup_active:
            # A rejected line search retains the active set and must not age a
            # floor-stability counter as though a lambda update were accepted.
            active_names = set(estimated_names) - frozen_names
            managed_cleanup_active_history.append(sorted(managed_cleanup_names - frozen_names))
            managed_cleanup_frozen_history.append(sorted(frozen_names))
        else:
            active_names = set(estimated_names)
            frozen_names.clear()

        # Step 8: Convergence check — strict tolerance OR objective plateau
        # Strict convergence still checks the accepted update across all
        # estimated components, including any names that were frozen earlier.
        changes = [
            abs(np.log(lambdas_new[pc.name]) - np.log(lambdas[pc.name]))
            for pc in all_pcs
            if pc.name in lambdas
            and pc.name in lambdas_new
            and lambdas[pc.name] > 0
            and lambdas_new[pc.name] > 0
        ]
        max_change = max(changes) if changes else 0.0

        # Plateau detection: objective flat and lambda changes small
        obj_rel_change = 0.0
        if len(objective_history) >= 2:
            obj_prev = objective_history[-2]
            obj_curr_val = objective_history[-1]
            obj_rel_change = abs(obj_curr_val - obj_prev) / max(abs(obj_curr_val), 1.0)

        # Converge on strict lambda tolerance
        strict_converged = candidate_accepted and max_change < reml_tol
        # An iteration still contracting the step is still buying precision;
        # the plateau may only classify a stalled endgame, never pre-empt
        # the strict road. With no previous accepted step there is no stall
        # evidence, so the plateau stays closed.
        if candidate_accepted:
            if accepted_change_window and accepted_change_window[-1] > 0.0:
                step_contraction_ratio = max_change / accepted_change_window[-1]
            accepted_change_window.append(max_change)
            del accepted_change_window[: -(_SCOP_EFS_PLATEAU_MIN_STALLED_ITERS + 1)]
        steps_stalled = _scop_plateau_steps_stalled(accepted_change_window, step_contraction_ratio)
        if managed_cleanup_active:
            plateau_converged = (
                candidate_accepted
                and n_reml_iter >= 3
                and steps_stalled
                and _multi_scop_discrete_plateau_converged(
                    obj_rel_change=obj_rel_change,
                    lambdas_old=lambdas,
                    lambdas_new=lambdas_new,
                    active_names=active_names,
                )
            )
        else:
            plateau_converged = (
                candidate_accepted
                and n_reml_iter >= 3
                and steps_stalled
                and obj_rel_change < 1e-6
                and max_change < 0.01
            )

        if verbose:
            lam_str = ", ".join(f"{pc.name}={lambdas_new[pc.name]:.4g}" for pc in all_pcs)
            print(
                f"  SCOP REML iter={n_reml_iter}  step={step_kind}  max_change={max_change:.6f}"
                f"  obj_rel={obj_rel_change:.2e}  lambdas=[{lam_str}]"
            )

        if debug_recorder is not None and getattr(debug_recorder, "enabled_level", 0) >= 2:
            debug_recorder.append_jsonl(
                "reml",
                {
                    "iteration": n_reml_iter,
                    "objective_before": float(obj_curr),
                    "objective_after": float(obj_after),
                    "lambda_max_delta": float(max_change),
                    "objective_relative_change": float(obj_rel_change),
                    "strict_converged": bool(strict_converged),
                    "plateau_converged": bool(plateau_converged),
                    "plateau_steps_stalled": bool(steps_stalled),
                    "candidate_accepted": bool(candidate_accepted),
                    "outer_step": step_kind,
                    "candidate_backoff_alpha": rescue_alpha,
                    "candidate_backoff_endorsed": rescue_endorsed,
                    "estimated_names": sorted(estimated_names),
                    "active_names": sorted(active_names),
                    "frozen_names": sorted(frozen_names),
                    "lambdas": {name: float(value) for name, value in lambdas_new.items()},
                },
            )

        lambda_history.append(lambdas_new.copy())

        # A rescued mode was chosen for certifiability, not objective merit,
        # and one that no objective gate ever endorsed must not be published
        # as stalled *or* converged.  The message is deliberately the
        # pre-backoff one: identical input, identical observable failure; the
        # payload row above carries candidate_backoff_alpha and
        # candidate_backoff_endorsed for anyone debugging which stage stalled.
        if rescue_endorsed is False:
            raise ObservedModeNotConvergedError(
                "SCOP REML candidate did not converge to a coefficient mode"
            )

        if not candidate_accepted:
            termination_reason = "line_search_stalled"
            lambdas = lambdas_new
            break

        if strict_converged or plateau_converged:
            converged = True
            termination_reason = "lambda_tolerance" if strict_converged else "objective_plateau"
            lambdas = lambdas_new
            break

        # Step 9: Warm start for next iteration
        lambdas = lambdas_new
        warm_beta = retained_mode.result.beta.copy()
        warm_intercept = float(retained_mode.result.intercept)
        warm_scop_states = retained_mode.scop_states if retained_mode.scop_states else None
        # Never read today -- the candidate site only fires on iteration 1 --
        # but kept current so a future loop shape cannot inherit a stale origin.
        step_origin = retained_mode

    if restart_with_efs:
        # The guard fired: rerun the search from the bootstrap with EFS steps,
        # in the iterations left. A fresh call reproduces the EFS search
        # exactly (its own fit context and centring state), so the answer is
        # the one ``_outer_step="efs"`` returns when the iterations left cover
        # that search; with fewer it stops at its own cap, as ``max_reml_iter``,
        # where the EFS search would have gone on. The Newton iterations cost
        # at most one candidate and three trial fits each on top of it.
        if verbose:
            print(
                f"  SCOP REML: Newton step rejected at iteration {n_reml_iter}; "
                "restarting the search with EFS steps"
            )
        restarted = optimize_scop_efs_reml(
            dm,
            distribution,
            link,
            groups,
            y,
            sample_weight,
            offset_arr,
            start_lambdas,
            estimated_names,
            weight_semantics=weight_semantics,
            max_reml_iter=max_reml_iter - n_reml_iter,
            reml_tol=reml_tol,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
            verbose=verbose,
            reml_penalties=reml_penalties,
            convergence=convergence,
            _scop_joint=_scop_joint,
            debug_recorder=debug_recorder,
            warm_lambdas=warm_lambdas,
            _aitken=_aitken,
            _outer_step="efs",
        )
        return replace(
            restarted,
            n_reml_iter=n_reml_iter + restarted.n_reml_iter,
            lambda_history=lambda_history + list(restarted.lambda_history),
            inner_iter_history=inner_iter_history + list(restarted.inner_iter_history or []),
            objective_history=objective_history + list(restarted.objective_history or []),
            scop_step_norms=(scop_step_norms_history + list(restarted.scop_step_norms or []))
            or None,
            scop_fisher_fallbacks=total_fisher_fallbacks + restarted.scop_fisher_fallbacks,
            scop_outer_steps=outer_steps + list(restarted.scop_outer_steps or []),
            scop_newton_fallback=newton_fallback,
            scop_newton_fallback_iter=newton_fallback_iter,
        )

    # -- Terminal mode --
    # An accepted line-search state has already paid for both its converged
    # coefficient fit and its coherent LAML evaluation.  Reuse it directly;
    # the fallback is needed only when the outer loop did not run.
    final_mode = retained_mode
    if final_mode is None or final_mode.lambdas != lambdas:
        if current_mode is not None and current_mode.lambdas == lambdas:
            final_mode = current_mode
        else:
            final_mode = _fit_scop_reml_mode(
                fit_context,
                lambdas,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                scop_state_init=warm_scop_states,
                phase="final",
                reml_iteration=n_reml_iter,
                require_converged=True,
            )
            if final_mode is None:
                raise ObservedModeNotConvergedError(
                    "SCOP REML final fit did not converge to a coefficient mode"
                )

    final_result = _finalize_scop_reml_mode(fit_context, final_mode)
    final_scop_states = final_mode.scop_states
    final_all_pcs = final_mode.penalty_components
    final_evaluation = final_mode.evaluation

    return REMLResult(
        lambdas=lambdas,
        pirls_result=final_result,
        n_reml_iter=n_reml_iter,
        converged=converged,
        lambda_history=lambda_history,
        reml_penalties=final_all_pcs,
        scop_states=final_scop_states if final_scop_states else None,
        objective=float(final_evaluation.value),
        inner_iter_history=inner_iter_history,
        objective_history=objective_history,
        curvature_source=final_mode.curvature_source,
        termination_reason=termination_reason,
        scop_step_norms=scop_step_norms_history if scop_step_norms_history else None,
        scop_fisher_fallbacks=total_fisher_fallbacks,
        managed_cleanup_names=sorted(managed_cleanup_names) if managed_cleanup_names else None,
        managed_cleanup_frozen_names=sorted(frozen_names) if managed_cleanup_active else None,
        managed_cleanup_freeze_iter=managed_cleanup_freeze_iter,
        managed_cleanup_active_history=(
            managed_cleanup_active_history if managed_cleanup_active_history else None
        ),
        managed_cleanup_frozen_history=(
            managed_cleanup_frozen_history if managed_cleanup_frozen_history else None
        ),
        tweedie_scale_data=tweedie_scale_data,
        flat_components=sorted(flat_names),
        scop_outer_steps=outer_steps,
        scop_newton_fallback=newton_fallback,
        scop_newton_fallback_iter=newton_fallback_iter,
    )
