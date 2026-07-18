"""PIRLS solver with pluggable penalty proximal operators."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._fit_trace import TraceRun
from superglm.distributions import _VARIANCE_FLOOR, Distribution, initial_mean
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    GroupMatrix,
)
from superglm.links import Link
from superglm.penalties.base import Penalty, penalty_can_zero_groups, penalty_targets_group
from superglm.solvers.centered_system import (
    build_centered_system,
    grouped_augmented_factor,
    grouped_weighted_factor,
)
from superglm.solvers.irls_state import _evaluate_irls_state, _IRLSState, _select_irls_trial
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankInfo,
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
)
from superglm.types import GroupSlice

logger = logging.getLogger(__name__)


def _positive_working_weight_stats(W: NDArray) -> tuple[float, float, float]:
    """Return positive W minimum, maximum, and ratio, excluding zero-weight rows."""
    positive = W[W > 0]
    if positive.size == 0:
        return float("nan"), float("nan"), float("inf")

    positive_min = float(np.min(positive))
    positive_max = float(np.max(positive))
    if not np.isfinite(positive_max):
        ratio = float("inf")
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = float(np.divide(positive_max, positive_min))
    return positive_min, positive_max, ratio


@dataclass
class IterationDiagnostics:
    """Per-iteration IRLS diagnostics for debugging convergence issues."""

    iteration: int
    deviance: float
    w_min: float
    w_max: float
    w_ratio: float
    mu_min: float
    mu_max: float
    eta_min: float
    eta_max: float
    intercept: float
    step_halvings: int
    # Indices of the 5 observations with largest/smallest W
    top_w_indices: NDArray  # (5,) int
    bottom_w_indices: NDArray  # (5,) int
    # Condition estimate and SVD fallback flag (direct solver only)
    cond_estimate: float | None = None
    used_svd_fallback: bool | None = None
    raw_w_min: float | None = None
    raw_w_max: float | None = None
    raw_w_ratio: float | None = None
    eta_min_unclipped: float | None = None
    eta_max_unclipped: float | None = None
    eta_clipped: bool | None = None
    working_mu_min: float | None = None
    working_mu_max: float | None = None
    working_eta_min: float | None = None
    working_eta_max: float | None = None
    working_eta_min_unclipped: float | None = None
    working_eta_max_unclipped: float | None = None
    working_eta_clipped: bool | None = None
    step_rejected: bool = False
    rank_truncated: bool | None = None
    trials_attempted: int = 1
    accepted_alpha: float = 1.0
    base_state_id: int | None = None
    proposal_state_id: int | None = None
    committed_state_id: int | None = None
    evaluation_id: int | None = None
    state_space: str = "solver"
    basis_id: int | None = None
    convergence_criterion: str | None = None
    convergence_value: float | None = None
    convergence_tolerance: float | None = None
    termination_reason: str | None = None


@dataclass
class PIRLSResult:
    beta: NDArray
    intercept: float
    n_iter: int
    deviance: float
    converged: bool
    phi: float
    effective_df: float
    iteration_log: list[IterationDiagnostics] | None = None
    log_det_H: float | None = None  # log|X'WX + S| from _safe_decompose_H  # noqa: N815
    rank_info: RankInfo | None = None
    state_id: int | None = None
    evaluation_id: int | None = None
    state_space: str = "solver"
    basis_id: int | None = None
    termination_reason: str | None = None


def _compute_group_hessians(
    gms: list[GroupMatrix],
    W: NDArray,
) -> tuple[list[float], list[NDArray]]:
    """Per-group Lipschitz constants and regularised Cholesky factors.

    Returns (L_groups, chol_groups) where:
    - L_groups[g] = max eigenvalue of X_g' diag(W) X_g
    - chol_groups[g] = lower Cholesky factor of (H_g + eps*I)

    For typical group sizes (p_g <= 20) this is trivially cheap.
    Total cost is O(n * p) across all groups.
    """
    L_groups: list[float] = []
    chol_groups: list[NDArray] = []
    for gm in gms:
        gram = gm.gram(W)
        L_g = max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
        L_groups.append(L_g)
        # Regularise: SSP reparametrisation can leave near-singular or
        # numerically-negative-definite Hessians (eigenvalues ≈ -1e-13).
        # eps = 1e-4 * L_g keeps condition number ≤ 1e4, losing ≤ 4 digits.
        eps = max(1e-4 * L_g, 1e-8)
        gram[np.diag_indices_from(gram)] += eps
        chol_groups.append(np.linalg.cholesky(gram))
    return L_groups, chol_groups


def _fit_pirls_inner(
    dm: DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    penalty: Penalty,
    offset: NDArray,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter_outer: int = 100,
    max_iter_inner: int = 5,
    tol: float = 1e-6,
    active_set: bool = False,
    lambda2: float | dict[str, float] = 0.0,
    record_diagnostics: bool = False,
    convergence: str = "deviance",
    S_override: NDArray | None = None,
    trace_run: TraceRun | None = None,
    trace_basis_id: int | None = None,
    trace_purpose: str = "fit",
) -> PIRLSResult:
    """Single-pass PIRLS fit with proximal Newton BCD inner solver."""
    n, p = dm.shape
    beta = beta_init.copy() if beta_init is not None else np.zeros(p)
    iteration_log: list[IterationDiagnostics] = [] if record_diagnostics else []

    # Initialize intercept
    if intercept_init is not None:
        intercept = intercept_init
    else:
        mu0 = initial_mean(y, weights, family)
        intercept = float(link.link(np.atleast_1d(mu0))[0])

    gms = dm.group_matrices
    n_groups = len(groups)
    can_zero_groups = penalty_can_zero_groups(penalty)

    t_total = time.perf_counter()
    t_lipschitz_total = 0.0
    t_inner_total = 0.0
    total_inner_iters = 0
    total_groups_skipped = 0

    trace_enabled = trace_run is not None and trace_run.enabled
    if trace_enabled and trace_basis_id is None:
        assert trace_run is not None
        trace_basis_id = trace_run.next_basis_id()
    resolved_lambdas: tuple[tuple[str, object], ...]
    if not trace_enabled:
        resolved_lambdas = ()
    elif isinstance(lambda2, dict):
        resolved_lambdas = tuple(
            [("selection", penalty.lambda1)]
            + [(f"smooth:{name}", float(value)) for name, value in sorted(lambda2.items())]
        )
    else:
        resolved_lambdas = (
            ("selection", penalty.lambda1),
            ("smooth", float(lambda2)),
        )

    def evaluate_state(
        beta_values: NDArray,
        intercept_value: float,
        *,
        phase: str,
        outer_iteration: int,
        alpha: float | None = None,
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
            state_id=state_id,
            evaluation_id=evaluation_id,
            basis_id=trace_basis_id,
            lambdas=resolved_lambdas,
        )
        if trace_enabled:
            assert trace_run is not None
            trace_run.emit_lazy(
                "evaluation",
                lambda: {
                    "state_id": state.state_id,
                    "evaluation_id": state.evaluation_id,
                    "solver": "pirls",
                    "phase": phase,
                    "outer_iteration": outer_iteration,
                    "accepted_alpha": alpha,
                    "state_space": state.state_space,
                    "basis_id": state.basis_id,
                    "lambdas": state.lambdas,
                    "dispersion": state.dispersion,
                    "intercept": state.intercept,
                    "deviance": state.deviance,
                },
                channel="pirls",
                purpose=trace_purpose,
                authoritative=False,
            )
        return state

    def emit_state_commit(
        state: _IRLSState,
        *,
        outer_iteration: int,
        fit_converged: bool,
        convergence_criterion: str | None,
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
                "solver": "pirls",
                "phase": "initial" if outer_iteration == 0 else "outer",
                "outer_iteration": outer_iteration,
                "state_space": state.state_space,
                "basis_id": state.basis_id,
                "lambdas": state.lambdas,
                "dispersion": state.dispersion,
                "intercept": state.intercept,
                "deviance": state.deviance,
                "fit_converged": fit_converged,
                "convergence_criterion": convergence_criterion,
                "convergence_value": convergence_value,
                "convergence_tolerance": tol,
                "termination_reason": termination_reason,
            },
            channel="pirls",
            purpose=trace_purpose,
        )

    # Freeze the fit-entry state so every trial is evaluated from fixed endpoints.
    committed = evaluate_state(beta, intercept, phase="initial", outer_iteration=0)
    emit_state_commit(
        committed,
        outer_iteration=0,
        fit_converged=False,
        convergence_criterion=None,
        convergence_value=None,
        termination_reason=None,
    )
    dev_prev = committed.deviance
    if not np.isfinite(dev_prev):
        dev_prev = np.inf
    converged = False
    max_halving = 5  # max step-halving attempts per outer iteration
    for outer in range(max_iter_outer):
        t_outer_start = time.perf_counter()

        beta_prev = committed.beta
        intercept_prev = committed.intercept
        beta = committed.beta.copy()
        intercept = committed.intercept

        # Current predictions are the complete retained snapshot.
        eta_unclipped = committed.eta_unclipped
        eta = committed.eta
        mu = committed.mu

        # Working weights and response (PIRLS)
        V = family.variance(mu)
        V = np.maximum(V, _VARIANCE_FLOOR)
        dmu_deta = link.deriv_inverse(eta)
        W = weights * dmu_deta**2 / V
        z = eta + (y - mu) / dmu_deta

        # Per-group Hessians and Lipschitz constants
        t0 = time.perf_counter()
        L_groups, chol_groups = _compute_group_hessians(gms, W)
        t_lipschitz_total += time.perf_counter() - t0

        # Initialize residual
        r = z - dm.matvec(beta) - intercept - offset

        # Active set: track which groups can be skipped.
        # A group is inactive if beta_g == 0 AND ||grad_g|| < lambda1 * w_g
        # (KKT optimality for zeroed group).  First inner iter is always
        # a full sweep; subsequent iters skip inactive groups.
        group_active = [True] * n_groups

        # Inner loop: proximal Newton block coordinate descent
        t_inner_start = time.perf_counter()
        for inner in range(max_iter_inner):
            # Periodic residual refresh to avoid float drift
            if inner > 0 and inner % 5 == 0:
                r = z - dm.matvec(beta) - intercept - offset

            beta_before = beta.copy()

            # Update intercept (closed form, unpenalised)
            delta_int: float = float(np.sum(W * r) / np.sum(W))
            intercept += delta_int
            r -= delta_int

            # BCD cycle over groups (Newton step + prox)
            for gi, (gm, g, L_g, chol_g) in enumerate(zip(gms, groups, L_groups, chol_groups)):
                # Active set: skip groups confirmed inactive on previous sweep
                if active_set and inner > 0 and not group_active[gi]:
                    total_groups_skipped += 1
                    continue

                bg_old = beta[g.sl].copy()

                grad_g = -gm.rmatvec(W * r)
                # Newton direction via Cholesky solve: H_g^{-1} grad_g
                newton_dir = scipy.linalg.cho_solve(
                    (chol_g, True),
                    grad_g,
                )
                step_g = 1.0 / L_g
                bg_cand = bg_old - newton_dir
                bg_new = penalty.prox_group(bg_cand, g, step_g)

                d = bg_new - bg_old
                if np.any(d != 0):
                    r -= gm.matvec(d)
                    beta[g.sl] = bg_new

                # Active set: check KKT for zeroed groups after the update
                if active_set:
                    lam = penalty.lambda1 if penalty.lambda1 is not None else 0.0
                    if not penalty_targets_group(penalty, g):
                        lam = 0.0
                    if np.linalg.norm(bg_new) < 1e-12:
                        # Group is zero — check if gradient is below threshold
                        # Use the gradient *after* the update (recompute cheaply)
                        grad_after = -gm.rmatvec(W * r)
                        kkt_thr = lam * g.weight * 0.9  # safety margin
                        group_active[gi] = np.linalg.norm(grad_after) >= kkt_thr
                    else:
                        group_active[gi] = True

            # Check inner convergence
            change: float = float(np.max(np.abs(beta - beta_before)))
            if change < tol * 0.01:
                break

        inner_iters = inner + 1
        total_inner_iters += inner_iters
        t_inner_total += time.perf_counter() - t_inner_start

        proposal = evaluate_state(
            beta,
            intercept,
            phase="proposal",
            outer_iteration=outer + 1,
            alpha=1.0,
        )
        trial_cache: dict[float, _IRLSState] = {1.0: proposal}

        def evaluate_trial(alpha: float) -> _IRLSState:
            beta_trial = committed.beta + alpha * (proposal.beta - committed.beta)
            intercept_trial = committed.intercept + alpha * (
                proposal.intercept - committed.intercept
            )
            candidate = evaluate_state(
                beta_trial,
                intercept_trial,
                phase="line_search_trial",
                outer_iteration=outer + 1,
                alpha=alpha,
            )
            trial_cache[alpha] = candidate
            return candidate

        decision = _select_irls_trial(
            committed=committed,
            proposal=proposal,
            evaluate_state=evaluate_trial,
            max_halving=max_halving,
        )
        retained = committed if decision.step_rejected else trial_cache[decision.alpha]
        beta = retained.beta.copy()
        intercept = retained.intercept
        eta_new_unclipped = retained.eta_unclipped
        eta_new = retained.eta
        mu_new = retained.mu
        dev = retained.deviance
        n_halvings = decision.step_halvings
        step_rejected = decision.step_rejected

        convergence_criterion = convergence
        if step_rejected or not np.isfinite(dev):
            convergence_value = float("inf")
            iteration_converged = False
        elif convergence == "coefficients":
            coef_change = float(np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta))))
            convergence_value = max(
                coef_change,
                abs(intercept - intercept_prev) / max(1.0, abs(intercept)),
            )
            iteration_converged = convergence_value < tol
        else:
            convergence_value = abs(dev - dev_prev) / (abs(dev_prev) + 1.0)
            iteration_converged = convergence_value < tol

        if step_rejected:
            termination_reason = "step_rejected"
        elif not np.isfinite(dev):
            termination_reason = "nonfinite_deviance"
        elif iteration_converged:
            termination_reason = "converged"
        elif outer + 1 == max_iter_outer:
            termination_reason = "max_iter"
        else:
            termination_reason = "continue"

        if trace_enabled:
            assert trace_run is not None
            trace_run.emit_lazy(
                "step_decision",
                lambda: {
                    "solver": "pirls",
                    "outer_iteration": outer + 1,
                    "base_state_id": committed.state_id,
                    "proposal_state_id": proposal.state_id,
                    "committed_state_id": retained.state_id,
                    "accepted_alpha": decision.alpha,
                    "step_halvings": decision.step_halvings,
                    "trials_attempted": decision.trials_attempted,
                    "step_rejected": decision.step_rejected,
                    "fit_converged": iteration_converged,
                    "convergence_criterion": convergence_criterion,
                    "convergence_value": convergence_value,
                    "convergence_tolerance": tol,
                    "termination_reason": termination_reason,
                },
                channel="pirls",
                purpose=trace_purpose,
            )
        emit_state_commit(
            retained,
            outer_iteration=outer + 1,
            fit_converged=iteration_converged,
            convergence_criterion=convergence_criterion,
            convergence_value=convergence_value,
            termination_reason=termination_reason,
        )
        if n_halvings:
            logger.info(
                "  PIRLS outer=%d: accepted step fraction %.5g after %d halvings, dev=%.2e",
                outer + 1,
                decision.alpha,
                n_halvings,
                dev,
            )

        # Warn on extreme working weight range (helps diagnose bad data)
        positive_w_min, positive_w_max, w_ratio = _positive_working_weight_stats(W)
        if w_ratio > 1e12:
            logger.debug(
                f"PIRLS outer={outer + 1}: extreme W ratio {w_ratio:.1e} "
                f"(positive W range [{positive_w_min:.2e}, {positive_w_max:.2e}])"
            )

        # Record per-iteration diagnostics
        if record_diagnostics:
            k = min(5, n)
            top_idx = np.argpartition(W, -k)[-k:]
            bot_idx = np.argpartition(W, k)[:k]
            working_eta_clipped = bool(
                float(np.min(eta_unclipped)) < float(np.min(eta))
                or float(np.max(eta_unclipped)) > float(np.max(eta))
            )
            eta_clipped = bool(
                float(np.min(eta_new_unclipped)) < float(np.min(eta_new))
                or float(np.max(eta_new_unclipped)) > float(np.max(eta_new))
            )
            iteration_log.append(
                IterationDiagnostics(
                    iteration=outer + 1,
                    deviance=dev,
                    w_min=float(W.min()),
                    w_max=float(W.max()),
                    w_ratio=w_ratio,
                    mu_min=float(mu_new.min()),
                    mu_max=float(mu_new.max()),
                    eta_min=float(eta_new.min()),
                    eta_max=float(eta_new.max()),
                    intercept=intercept,
                    step_halvings=n_halvings,
                    top_w_indices=top_idx[np.argsort(W[top_idx])[::-1]],
                    bottom_w_indices=bot_idx[np.argsort(W[bot_idx])],
                    raw_w_min=float(W.min()),
                    raw_w_max=float(W.max()),
                    raw_w_ratio=w_ratio,
                    eta_min_unclipped=float(np.min(eta_new_unclipped)),
                    eta_max_unclipped=float(np.max(eta_new_unclipped)),
                    eta_clipped=eta_clipped,
                    working_mu_min=float(mu.min()),
                    working_mu_max=float(mu.max()),
                    working_eta_min=float(eta.min()),
                    working_eta_max=float(eta.max()),
                    working_eta_min_unclipped=float(np.min(eta_unclipped)),
                    working_eta_max_unclipped=float(np.max(eta_unclipped)),
                    working_eta_clipped=working_eta_clipped,
                    step_rejected=step_rejected,
                    trials_attempted=decision.trials_attempted,
                    accepted_alpha=decision.alpha,
                    base_state_id=committed.state_id,
                    proposal_state_id=proposal.state_id,
                    committed_state_id=retained.state_id,
                    evaluation_id=retained.evaluation_id,
                    state_space=retained.state_space,
                    basis_id=retained.basis_id,
                    convergence_criterion=convergence_criterion,
                    convergence_value=convergence_value,
                    convergence_tolerance=tol,
                    termination_reason=termination_reason,
                )
            )

        t_outer_elapsed = time.perf_counter() - t_outer_start
        logger.info(
            f"  outer={outer + 1:3d}  bcd_cycles={inner_iters:4d}  "
            f"dev={dev:12.1f}  delta={abs(dev - dev_prev) / (abs(dev_prev) + 1):10.2e}  "
            f"time={t_outer_elapsed:.3f}s"
        )

        if step_rejected:
            logger.warning(
                "PIRLS rejected all trial steps at outer=%d; restored committed state "
                "(committed dev=%.6g, proposal dev=%.6g, trials=%s)",
                outer + 1,
                committed.deviance,
                proposal.deviance,
                {alpha: state.deviance for alpha, state in trial_cache.items()},
            )
            break

        if not np.isfinite(dev):
            logger.warning(f"PIRLS non-finite deviance at outer={outer + 1}: dev={dev:.2e}")
            break

        if iteration_converged:
            converged = True
            break
        committed = retained
        dev_prev = dev

    t_elapsed = time.perf_counter() - t_total
    logger.info(
        f"  PIRLS done: {outer + 1} outer iters, {total_inner_iters} total BCD cycles, "
        f"{t_elapsed:.2f}s total"
    )
    extra = ""
    if active_set:
        total_group_updates = total_inner_iters * n_groups
        extra = f"  groups_skipped={total_groups_skipped}/{total_group_updates}"
    logger.info(
        f"  Breakdown: group_lipschitz={t_lipschitz_total:.2f}s  bcd_cycles={t_inner_total:.2f}s"
        + extra
    )

    # Effective df: exact hat-matrix trace when lambda2 > 0 (smoothing active),
    # Breheny-Huang (2009) group lasso formula when lambda2 = 0.
    has_smoothing = (isinstance(lambda2, dict) and any(v > 0 for v in lambda2.values())) or (
        not isinstance(lambda2, dict) and lambda2 > 0
    )

    from superglm.reml.penalty_algebra import build_penalty_matrix

    # Selection is derived from the final retained coefficients.  A proposal
    # may have been step-halved or rejected, so its proximal state is not an
    # accepted source of rank and inference metadata.
    group_selected = [
        not can_zero_groups
        or not penalty_targets_group(penalty, group)
        or bool(np.any(beta[group.sl] != 0.0))
        for group in groups
    ]
    selected_columns_list: list[int] = []
    selected_groups: list[GroupSlice] = []
    selected_gms: list[GroupMatrix] = []
    selected_group_names: list[str] = []
    selected_offset = 0
    for is_selected, gm, group in zip(group_selected, gms, groups, strict=True):
        if not is_selected:
            continue
        selected_columns_list.extend(range(group.start, group.end))
        selected_group_names.append(group.name)
        selected_gms.append(gm)
        selected_groups.append(
            GroupSlice(
                name=group.name,
                start=selected_offset,
                end=selected_offset + group.size,
                weight=group.weight,
                penalized=group.penalized,
                feature_name=group.feature_name,
                subgroup_type=group.subgroup_type,
            )
        )
        selected_offset += group.size

    selected_columns = np.asarray(selected_columns_list, dtype=int)
    selected_dm = DesignMatrix(selected_gms, n=n, p=len(selected_columns))
    if has_smoothing:
        if S_override is not None:
            selected_penalty = S_override[np.ix_(selected_columns, selected_columns)]
        else:
            selected_penalty = build_penalty_matrix(
                selected_gms,
                selected_groups,
                lambda2,
                len(selected_columns),
            )
    else:
        selected_penalty = np.zeros((len(selected_columns), len(selected_columns)))

    V_final = np.maximum(family.variance(mu_new), _VARIANCE_FLOOR)
    dmu_deta_final = link.deriv_inverse(eta_new)
    W_final = weights * dmu_deta_final**2 / V_final
    z_final = eta_new + (y - mu_new) / dmu_deta_final
    centered = build_centered_system(
        dm=selected_dm,
        W=W_final,
        z_off=z_final - offset,
        penalty=selected_penalty,
    )
    data_rank = decompose_gram(centered.data_gram)
    if needs_factor_certification(data_rank):
        certified = decompose_factor(
            grouped_weighted_factor(
                selected_dm,
                W_final,
                center=centered.mean_x,
            )
        )
        if certified.rank != data_rank.rank:
            data_rank = certified
    augmented_rank = data_rank if not np.any(selected_penalty) else decompose_gram(centered.hessian)
    if needs_factor_certification(augmented_rank):
        certified = decompose_factor(
            grouped_augmented_factor(
                selected_dm,
                W_final,
                selected_penalty,
                center=centered.mean_x,
            )
        )
        if certified.rank != augmented_rank.rank:
            augmented_rank = certified
    raw_gram, _, _, _ = centered.raw_weighted_moments()
    coefficient_rank = decompose_gram(raw_gram + selected_penalty)
    if needs_factor_certification(coefficient_rank):
        certified = decompose_factor(
            grouped_augmented_factor(selected_dm, W_final, selected_penalty)
        )
        if certified.rank != coefficient_rank.rank:
            coefficient_rank = certified
    feature_edf = np.zeros(p)
    group_edf = {group.name: 0.0 for group in groups}

    if has_smoothing:
        selected_edf = np.diag(augmented_rank.pseudo_inverse() @ centered.data_gram).copy()
        selected_edf[np.abs(selected_edf) < 100.0 * np.finfo(float).eps] = 0.0
        feature_edf[selected_columns] = selected_edf
        for selected_group, original_group in zip(
            selected_groups,
            (group for selected, group in zip(group_selected, groups, strict=True) if selected),
            strict=True,
        ):
            group_edf[original_group.name] = float(np.sum(selected_edf[selected_group.sl]))
    else:
        # Preserve Breheny-Huang (2009) group-lasso EDF allocation.
        lam = penalty.lambda1 if penalty.lambda1 is not None else 0.0
        for is_selected, group in zip(group_selected, groups, strict=True):
            if not is_selected:
                continue
            norm_g = float(np.linalg.norm(beta[group.sl]))
            if not penalty_targets_group(penalty, group) or not can_zero_groups:
                df_group = float(group.size)
            else:
                shrink = min(1.0, lam * group.weight / max(norm_g, 1e-300))
                df_group = float(group.size - (group.size - 1) * shrink)
            group_edf[group.name] = df_group
            feature_edf[group.sl] = df_group / group.size

    mean_x = np.zeros(p)
    mean_x[selected_columns] = centered.mean_x
    selected_columns.setflags(write=False)
    mean_x.setflags(write=False)
    feature_edf.setflags(write=False)
    rank_info = RankInfo(
        policy_version=SHARED_RANK_POLICY.version,
        coordinate_space="solver",
        selected_columns=selected_columns,
        selected_group_names=tuple(selected_group_names),
        sum_w=centered.sum_w,
        mean_x=mean_x,
        intercept_edf=1.0,
        data=data_rank,
        augmented=augmented_rank,
        coefficient=coefficient_rank,
        feature_edf=feature_edf,
        group_edf=group_edf,
        objective_loss=None,
    )
    p_eff = rank_info.total_edf

    # Pearson-based phi for estimated-scale families (Tweedie, Gamma, NB2).
    # SuperGLM's sample_weight follows the prior-weight convention, so the
    # residual d.f. correction is observation-count based (n - edf), while
    # the weights still scale the Pearson numerator.
    pearson_chi2 = float(np.sum(weights * (y - mu_new) ** 2 / V_final))
    df_resid = max(float(len(y)) - p_eff, 1)
    phi = pearson_chi2 / df_resid

    return PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=outer + 1,
        deviance=dev,
        converged=converged,
        phi=phi,
        effective_df=p_eff,
        iteration_log=iteration_log if record_diagnostics else None,
        rank_info=rank_info,
        state_id=retained.state_id,
        evaluation_id=retained.evaluation_id,
        state_space=retained.state_space,
        basis_id=retained.basis_id,
        termination_reason=termination_reason,
    )


def _wrap_dense_X(X: NDArray, groups: list[GroupSlice]) -> DesignMatrix:
    """Wrap a dense NDArray into a DesignMatrix for backward compatibility."""
    n, p = X.shape
    gms = [DenseGroupMatrix(X[:, g.sl]) for g in groups]
    return DesignMatrix(gms, n, p)


def fit_pirls(
    X: NDArray | DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    penalty: Penalty,
    offset: NDArray | None = None,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter_outer: int = 100,
    max_iter_inner: int = 5,
    tol: float = 1e-6,
    active_set: bool = False,
    lambda2: float | dict[str, float] = 0.0,
    record_diagnostics: bool = False,
    convergence: str = "deviance",
    S_override: NDArray | None = None,
    trace_run: TraceRun | None = None,
) -> PIRLSResult:
    """Fit a penalised GLM via PIRLS with proximal Newton BCD.

    If the penalty has a flavor (e.g. Adaptive), a two-stage fit is performed:
    1. Fit with uniform weights → beta_init
    2. Flavor adjusts group weights based on beta_init
    3. Refit with adjusted weights (warm started from stage 1)
    """
    if isinstance(X, DesignMatrix):
        dm = X
        n = dm.n
    else:
        dm = _wrap_dense_X(X, groups)
        n = X.shape[0]

    if offset is None:
        offset = np.zeros(n)

    trace_basis_id = (
        trace_run.next_basis_id() if trace_run is not None and trace_run.enabled else None
    )

    # Stage 1: initial fit
    result = _fit_pirls_inner(
        dm,
        y,
        weights,
        family,
        link,
        groups,
        penalty,
        offset,
        beta_init,
        intercept_init,
        max_iter_outer,
        max_iter_inner,
        tol,
        active_set,
        lambda2=lambda2,
        record_diagnostics=record_diagnostics,
        convergence=convergence,
        S_override=S_override,
        trace_run=trace_run,
        trace_basis_id=trace_basis_id,
        trace_purpose="initial_flavor_fit" if penalty.flavor is not None else "fit",
    )

    # Stage 2: if flavor, adjust weights and refit (warm-start both beta and intercept)
    if penalty.flavor is not None:
        adjusted_groups = penalty.flavor.adjust_weights(
            groups, result.beta, group_matrices=dm.group_matrices
        )
        result = _fit_pirls_inner(
            dm,
            y,
            weights,
            family,
            link,
            adjusted_groups,
            penalty,
            offset,
            beta_init=result.beta,
            intercept_init=result.intercept,
            max_iter_outer=max_iter_outer,
            max_iter_inner=max_iter_inner,
            tol=tol,
            active_set=active_set,
            lambda2=lambda2,
            record_diagnostics=record_diagnostics,
            convergence=convergence,
            S_override=S_override,
            trace_run=trace_run,
            trace_basis_id=trace_basis_id,
            trace_purpose="adjusted_flavor_fit",
        )

    return result
