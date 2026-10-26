"""What one direct IRLS fit discloses about itself besides its result (``_IRLSDirectRecorder``)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from superglm.solvers.irls_state import _IRLSState
from superglm.solvers.pirls import (
    IterationDiagnostics,
    TerminationReason,
    _extreme_weight_indices,
)
from superglm.solvers.structured import record_auto_backend_decision

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)


class _IRLSDirectRecorder:
    """What one direct IRLS fit discloses about itself besides its result.

    The trace (evaluations, state commits and step decisions, each payload built
    only when the trace is on), the per-iteration diagnostics and their extrema,
    the level-2 debug rows, the phase timers and profile counters, the stop rule's
    numbers and the backend disclosure.  It holds no reference to the fit: a method
    that reads the fit's state is handed the fit, so the recorder never keeps the
    fit's arrays alive.
    """

    __slots__ = (
        "trace_run trace_purpose trace_enabled trace_basis_id resolved_lambdas convergence tol"
        " profile record_diagnostics debug_recorder base_debug_context record_debug_rows"
        " capture_extrema iteration_log eta_clipped working_eta_clipped t_start t_elapsed"
        " t_working t_gram t_solve t_deviance t_eta t_deviance_eval"
    ).split()

    def __init__(self, fit: _IRLSDirectFit) -> None:
        trace_run, lambda2 = fit.trace_run, fit.lambda2
        trace_enabled = trace_run is not None and trace_run.enabled
        self.trace_basis_id = (
            trace_run.next_basis_id() if trace_enabled and trace_run is not None else None
        )
        if not trace_enabled:
            resolved_lambdas: tuple[tuple[str, object], ...] = ()
        elif isinstance(lambda2, dict):
            resolved_lambdas = tuple(
                (f"smooth:{name}", float(value)) for name, value in sorted(lambda2.items())
            )
        else:
            resolved_lambdas = (("smooth", float(lambda2)),)
        self.trace_run, self.trace_purpose = trace_run, fit.trace_purpose
        self.trace_enabled, self.resolved_lambdas = trace_enabled, resolved_lambdas
        self.convergence, self.tol = fit.convergence, fit.tol
        self.profile, self.record_diagnostics = fit.profile, fit.record_diagnostics
        self.debug_recorder = debug_recorder = fit.debug_recorder
        self.base_debug_context = dict(fit.debug_context or {})
        # Level 2 fixes the row schema for the whole fit, so snapshot it at fit entry.
        self.record_debug_rows = (
            debug_recorder is not None and getattr(debug_recorder, "enabled_level", 0) >= 2
        )
        # Strictly wider than ``record_diagnostics``, so the per-iteration extrema
        # this gates are always bound by the time a diagnostics row is built.
        self.capture_extrema = fit.record_diagnostics or self.record_debug_rows
        # Always an empty list; only ``record_diagnostics`` decides whether rows are
        # appended and whether it is published on the result.
        self.iteration_log: list[IterationDiagnostics] = []
        # Phase timing accumulators
        self.t_working = 0.0
        self.t_gram = 0.0
        self.t_solve = 0.0
        self.t_deviance = 0.0
        self.t_eta = 0.0
        self.t_deviance_eval = 0.0

    def count(self, key: str, n: int = 1) -> None:
        """Add ``n`` to the profile counter ``key`` when the caller keeps a profile."""
        profile = self.profile
        if profile is not None:
            profile[key] = profile.get(key, 0) + n

    def peak(self, key: str, value: float) -> None:
        """Keep the largest ``value`` under ``key`` when the caller keeps a profile."""
        profile = self.profile
        if profile is not None:
            profile[key] = max(float(profile.get(key, 0.0)), value)

    def emit_evaluation(
        self,
        state: _IRLSState,
        *,
        phase: str,
        iteration: int,
        alpha: float | None = None,
        enclosing_proposal_state_id: int | None = None,
        deviance_reused: bool = False,
    ) -> None:
        if not self.trace_enabled:
            return
        assert self.trace_run is not None
        self.trace_run.emit_lazy(
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
            purpose=self.trace_purpose,
            authoritative=False,
        )

    def emit_state_commit(
        self,
        state: _IRLSState,
        *,
        iteration: int,
        fit_converged: bool,
        convergence_value: float | None,
        termination_reason: TerminationReason | None,
        convergence_label: str | None = None,
        convergence_tolerance: float | None = None,
    ) -> None:
        if not self.trace_enabled:
            return
        assert self.trace_run is not None
        convergence, tol = self.convergence, self.tol
        self.trace_run.emit_lazy(
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
                "convergence_criterion": convergence
                if convergence_label is None
                else convergence_label,
                "convergence_value": convergence_value,
                "convergence_tolerance": tol
                if convergence_tolerance is None
                else convergence_tolerance,
                "termination_reason": termination_reason,
            },
            channel="pirls",
            purpose=self.trace_purpose,
        )

    def start(self) -> None:
        """The loop's clock starts: ``irls_total_s`` runs from here to ``stop``."""
        self.t_start = time.perf_counter()

    def iteration(self, fit: _IRLSDirectFit, it: int) -> None:
        """The iteration's step decision and state commit, extrema and diagnostics row."""
        committed, proposal_state, retained = fit.committed, fit.proposal_state, fit.retained
        decision, converged_this_iter = fit.decision, fit.converged_this_iter
        convergence_label, convergence_value = fit.convergence_label, fit.convergence_value
        convergence_tolerance = fit.convergence_tolerance
        termination_reason, working_rows = fit.termination_reason, fit.working_rows
        newton_score, W = fit.newton_score, fit.W
        if self.trace_enabled:
            assert self.trace_run is not None
            self.trace_run.emit_lazy(
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
                    "convergence_criterion": convergence_label,
                    "convergence_value": convergence_value,
                    "convergence_tolerance": convergence_tolerance,
                    "termination_reason": termination_reason,
                    "working_curvature": (
                        working_rows.curvature_source if newton_score is None else "observed"
                    ),
                },
                channel="pirls",
                purpose=self.trace_purpose,
            )
        if np.isfinite(fit.dev):
            self.emit_state_commit(
                retained,
                iteration=it + 1,
                fit_converged=converged_this_iter,
                convergence_value=convergence_value,
                termination_reason=termination_reason,
                convergence_label=convergence_label,
                convergence_tolerance=convergence_tolerance,
            )

        working_eta_clipped = False
        eta_clipped = False
        if self.capture_extrema:
            # Each of these is read again by the diagnostics row below, which
            # only runs when ``capture_extrema`` is true (see its definition).
            # Bind once: they are O(n) passes over arrays nothing mutates
            # between here and the append.
            working_eta_min = float(np.min(fit.working_eta))
            working_eta_max = float(np.max(fit.working_eta))
            working_eta_min_unclipped = float(np.min(fit.working_eta_unclipped))
            working_eta_max_unclipped = float(np.max(fit.working_eta_unclipped))
            eta_min = float(np.min(fit.eta))
            eta_max = float(np.max(fit.eta))
            eta_min_unclipped = float(np.min(fit.eta_unclipped))
            eta_max_unclipped = float(np.max(fit.eta_unclipped))
            working_eta_clipped = bool(
                working_eta_min_unclipped < working_eta_min
                or working_eta_max_unclipped > working_eta_max
            )
            eta_clipped = bool(eta_min_unclipped < eta_min or eta_max_unclipped > eta_max)

        # Record per-iteration diagnostics
        if self.record_diagnostics:
            top_idx, bot_idx = _extreme_weight_indices(W)
            w_min = float(W.min())
            w_max = float(W.max())
            self.iteration_log.append(
                IterationDiagnostics(
                    iteration=it + 1,
                    deviance=fit.dev,
                    w_min=w_min,
                    w_max=w_max,
                    w_ratio=fit.w_ratio,
                    mu_min=float(fit.mu.min()),
                    mu_max=float(fit.mu.max()),
                    eta_min=eta_min,
                    eta_max=eta_max,
                    intercept=fit.intercept,
                    step_halvings=fit.n_halvings,
                    top_w_indices=top_idx,
                    bottom_w_indices=bot_idx,
                    cond_estimate=fit._cond_est,
                    used_svd_fallback=fit._used_svd,
                    raw_w_min=w_min,
                    raw_w_max=w_max,
                    raw_w_ratio=fit.w_ratio,
                    eta_min_unclipped=eta_min_unclipped,
                    eta_max_unclipped=eta_max_unclipped,
                    eta_clipped=eta_clipped,
                    working_mu_min=float(fit.working_mu.min()),
                    working_mu_max=float(fit.working_mu.max()),
                    working_eta_min=working_eta_min,
                    working_eta_max=working_eta_max,
                    working_eta_min_unclipped=working_eta_min_unclipped,
                    working_eta_max_unclipped=working_eta_max_unclipped,
                    working_eta_clipped=working_eta_clipped,
                    step_rejected=fit.step_rejected,
                    rank_truncated=fit.rank_truncated,
                    trials_attempted=decision.trials_attempted,
                    accepted_alpha=decision.alpha,
                    base_state_id=committed.state_id,
                    proposal_state_id=proposal_state.state_id,
                    committed_state_id=retained.state_id,
                    evaluation_id=retained.evaluation_id,
                    state_space=retained.state_space,
                    basis_id=retained.basis_id,
                    convergence_criterion=convergence_label,
                    convergence_value=convergence_value,
                    convergence_tolerance=convergence_tolerance,
                    termination_reason=termination_reason,
                )
            )
        self.eta_clipped, self.working_eta_clipped = eta_clipped, working_eta_clipped

    def debug_row(self, fit: _IRLSDirectFit, it: int) -> None:
        """The iteration's level-2 debug row (``record_debug_rows``)."""
        self.debug_recorder.append_jsonl(
            "pirls",
            {
                **self.base_debug_context,
                "iteration": it + 1,
                "deviance": float(fit.dev),
                "deviance_relative_change": (
                    float(fit.dev_rel_change) if fit.dev_rel_change is not None else None
                ),
                "coefficient_change": float(fit.coef_change)
                if fit.coef_change is not None
                else None,
                "convergence": self.convergence,
                "converged": bool(fit.converged_this_iter),
                "w_min": float(fit.W.min()),
                "w_max": float(fit.W.max()),
                "w_ratio": float(fit.w_ratio),
                "mu_min": float(fit.mu.min()),
                "mu_max": float(fit.mu.max()),
                "eta_min_unclipped": float(np.min(fit.eta_unclipped)),
                "eta_max_unclipped": float(np.max(fit.eta_unclipped)),
                "eta_clipped": bool(self.eta_clipped),
                "eta_min": float(fit.eta.min()),
                "eta_max": float(fit.eta.max()),
                "working_mu_min": float(fit.working_mu.min()),
                "working_mu_max": float(fit.working_mu.max()),
                "working_eta_min_unclipped": float(np.min(fit.working_eta_unclipped)),
                "working_eta_max_unclipped": float(np.max(fit.working_eta_unclipped)),
                "working_eta_clipped": bool(self.working_eta_clipped),
                "working_eta_min": float(fit.working_eta.min()),
                "working_eta_max": float(fit.working_eta.max()),
                "step_halvings": int(fit.n_halvings),
                "trials_attempted": int(fit.decision.trials_attempted),
                "step_rejected": bool(fit.step_rejected),
                "base_state_id": fit.committed.state_id,
                "proposal_state_id": fit.proposal_state.state_id,
                "committed_state_id": fit.retained.state_id,
                "rank_truncated": fit.rank_truncated,
                "cond_estimate": float(fit._cond_est),
                "used_svd_fallback": bool(fit._used_svd),
                "has_scop": bool(fit._has_scop),
                "working_curvature": (
                    fit.working_rows.curvature_source if fit.newton_score is None else "observed"
                ),
            },
        )

    def stop(self, it: int) -> None:
        """The loop's clock stops; the log line reports the iterations and the time."""
        self.t_elapsed = t_elapsed = time.perf_counter() - self.t_start
        logger.info(f"  IRLS direct done: {it + 1} iters, {t_elapsed:.2f}s")

    def profile_fit(self, fit: _IRLSDirectFit, it: int) -> None:
        """The phase times, the call and iteration counts, and the stop rule's numbers."""
        profile, mode_bar = self.profile, fit.mode_bar
        _last_mode_residual = fit._last_mode_residual
        # Accumulate phase timing into the profile dict if provided
        if profile is not None:
            profile["irls_working_s"] = profile.get("irls_working_s", 0.0) + self.t_working
            profile["irls_gram_s"] = profile.get("irls_gram_s", 0.0) + self.t_gram
            profile["irls_solve_s"] = profile.get("irls_solve_s", 0.0) + self.t_solve
            profile["irls_deviance_s"] = profile.get("irls_deviance_s", 0.0) + self.t_deviance
            profile["irls_eta_s"] = profile.get("irls_eta_s", 0.0) + self.t_eta
            profile["irls_deviance_eval_s"] = (
                profile.get("irls_deviance_eval_s", 0.0) + self.t_deviance_eval
            )
            profile["irls_total_s"] = profile.get("irls_total_s", 0.0) + self.t_elapsed
            profile["irls_calls"] = profile.get("irls_calls", 0) + 1
            profile["irls_iters"] = profile.get("irls_iters", 0) + (it + 1)
            if fit._boundary_rows:
                profile["irls_mean_space_boundary_calls"] = (
                    profile.get("irls_mean_space_boundary_calls", 0) + 1
                )
            if fit._true_stop and fit._last_true_residual[0] is not None:
                # the true-score certificate decided the last iteration (Newton
                # steps): its score, floors and exclusions are the stop rule's
                true_residual = fit._last_true_residual[0]
                profile["irls_mode_rule"] = "mean_space_mode_score"
                profile["irls_mode_score"] = mode_bar * fit._true_stop_ratio
                profile["irls_mode_bar"] = mode_bar
                profile["irls_mode_floor_binding"] = true_residual.floor_binding
                profile["irls_mode_weakly_identified"] = tuple(
                    int(index) for index in np.flatnonzero(true_residual.excluded)
                )
            elif _last_mode_residual is not None:
                # §3.8, §3.9: the stop rule's last score, whether a derived floor
                # bound it, and the slopes it flagged weakly identified and kept.
                profile["irls_mode_score"] = mode_bar * _last_mode_residual.ratio()
                profile["irls_mode_bar"] = mode_bar
                profile["irls_mode_floor_binding"] = _last_mode_residual.floor_binding
                profile["irls_mode_weakly_identified"] = tuple(
                    int(index) for index in np.flatnonzero(_last_mode_residual.excluded)
                )

    def finalize_time(self, t0: float) -> None:
        """The finalisation's time since ``t0``, when the caller keeps a profile."""
        profile = self.profile
        if profile is not None:
            _t_finalize = time.perf_counter() - t0
            profile["irls_finalize_s"] = profile.get("irls_finalize_s", 0.0) + _t_finalize

    def disclose_backend(self, fit: _IRLSDirectFit, backend: str) -> None:
        """The backend the fit ran on, why, and the structured factor's certificate."""
        profile, structured_decision = self.profile, fit.structured_decision
        structured_factor, groups = fit.structured_factor, fit.groups
        if profile is not None:
            profile["direct_backend"] = backend
            profile["direct_fallback_reason"] = fit._direct_fallback_reason
            # Repeated inner solves stay quiet; the REML drivers own the INFO line.
            record_auto_backend_decision(profile, fit.direct_solve, structured_decision, log=False)
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
                    profile["structured_border_verification_decrements"] = (
                        border_certificate.decrements
                    )
                    profile["structured_border_trailing_bound"] = border_certificate.trailing_bound
                    profile["structured_border_deflated"] = border_certificate.deflated
                    profile["structured_weakly_identified"] = tuple(
                        getattr(structured_factor, "weakly_identified_slopes", ())
                    )
                profile["structured_minimum_local_diagonal"] = (
                    structured_factor.minimum_local_diagonal
                )
                profile["structured_schur_condition"] = structured_factor.schur_condition_estimate
