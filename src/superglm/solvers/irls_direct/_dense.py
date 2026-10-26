"""The dense step: the Gram path's centred normal equations, for a fit without linear constraints."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from superglm.solvers.irls_direct._centred import _system_offset_mean
from superglm.solvers.irls_direct._step import _IRLSDirectStep, _note_svd_fallback
from superglm.solvers.mode_score import centred_data_score
from superglm.solvers.rank import decompose_gram_if_authoritative

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit


class _IRLSDirectDenseStep(_IRLSDirectStep):
    """The fit's step on the Gram path when it has no linear constraint: Cholesky or a certified factor.

    Chosen in ``_IRLSDirectFit._setup_systems`` when the fit takes none of the
    QR, structured, SCOP and QP routes.  Each iteration forms the centred normal
    equations, the intercept profiled out, through the fit's centred system and
    its caches, and solves them by Cholesky where the Gram certifies its own
    rank, else through the factor certificate of the grouped augmented design
    (``certify_centered_factor``).  It keeps no reference to the fit, whose
    state each method is handed, and its rank metadata takes the module route,
    which may be deferred.
    """

    __slots__ = ()
    backend = "gram"

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        # Gram path: form X'WX via per-group gram, solve (p+1)×(p+1).
        _t0 = time.perf_counter()
        z_off = fit.z - fit.offset
        W, dm, newton_score, _state_center = fit.W, fit.dm, fit.newton_score, fit._state_center
        proposal_centred_intercept = fit.proposal_centred_intercept
        centered = fit.get_centered_system(W, z_off)
        if newton_score is not None:
            # the working mean of ``z_off`` is the centred intercept;
            # the score moves it by sum(u) / sum(W) and the slopes'
            # right-hand side by X~' u (the step itself is the increment
            # below; the stored system is the one it solved)
            centered = replace(
                centered,
                mean_z=centered.mean_z + float(np.sum(newton_score)) / centered.sum_w,
                rhs=centered.rhs + centred_data_score(dm, newton_score, centered.mean_x),
            )
        fit._last_working_centered = centered
        fit.recorder.t_gram += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        iteration_rank = decompose_gram_if_authoritative(centered.hessian)
        iteration_factor_rhs = None
        if iteration_rank is None:
            certification = fit.certify_centered_factor(
                centered,
                W,
                response=None if newton_score is not None else z_off - centered.mean_z,
            )
            certified = certification.decomposition
            if certification.transformed_rhs is None and newton_score is None:
                raise RuntimeError(  # pragma: no cover - invariant
                    "factor certification omitted its transformed RHS"
                )
            iteration_rank = certified
            iteration_factor_rhs = certification.transformed_rhs
            fit.used_rank_certification = True
        if newton_score is not None:
            # the increment, not the iterate: a direction the factor
            # truncates keeps the iterate's component instead of the
            # minimum-norm reset of solving for it (Pes & Rodriguez
            # 2021, eqs. 1.2-1.4)
            beta = fit.committed.beta + iteration_rank.solve(
                centred_data_score(dm, newton_score, centered.mean_x)
                - fit.penalty_matvec(fit.committed.beta)
            )
        else:
            beta = (
                iteration_rank.solve(centered.rhs)
                if iteration_factor_rhs is None
                else iteration_rank.solve_factor_rhs(iteration_factor_rhs)
            )
        intercept = centered.mean_z - float(centered.mean_x @ beta)
        _last_working_offset_mean = None
        if _state_center is not None:
            # the intercept about the state's centre, from the offset
            # of the working mean to it, formed on centred rows (the
            # system's pair, else ``centre_offset_mean``): no
            # raw-scale cancellation
            _last_working_offset_mean = _system_offset_mean(dm, W, centered, _state_center)
            proposal_centred_intercept = centered.mean_z - math.fsum(
                _last_working_offset_mean * beta
            )
            intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
        fit._cond_est = iteration_rank.pre_truncation_condition
        fit._held_rank[0] = iteration_rank
        fit._used_svd = iteration_rank.used_svd_fallback
        fit.rank_truncated = iteration_rank.rank_truncated
        if fit.convergence == "mode_score":
            fit._score_centre = (
                centered.mean_x,
                centered.sum_w,
                np.diag(centered.data_gram),
                (),
                _last_working_offset_mean,
            )
        fit.recorder.t_solve += time.perf_counter() - _t0
        fit.beta, fit.intercept = beta, intercept
        fit._last_working_offset_mean = _last_working_offset_mean
        fit.proposal_centred_intercept = proposal_centred_intercept
        _note_svd_fallback(fit)
