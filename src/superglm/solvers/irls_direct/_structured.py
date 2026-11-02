"""The structured step: a random-effect chain, ``fs`` or ``sz`` term solved in block form, never as a dense Gram."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm._blas_threads import keep_narrow_cap
from superglm.group_matrix import DesignMatrix
from superglm.solvers._structured.assembly import (
    build_augmented_structured_factor,
    build_penalized_structured_operator,
    profile_structured_factor,
    solve_augmented_normal_equations,
)
from superglm.solvers._structured.balance_tree import (
    SumToZeroLeafSystem,
    SumToZeroPenalizedOperator,
    SumToZeroTreeFactor,
)
from superglm.solvers._structured.block_leaves import (
    FactorSmoothLeafSystem,
    FactorSmoothPenalizedOperator,
    factor_smooth_prior_statistics,
)
from superglm.solvers._structured.geometry import (
    cancelled_column_row_norms,
)
from superglm.solvers._structured.layout import (
    FactorSmoothLeafLayout,
    get_structured_layout,
)
from superglm.solvers._structured.moments import (
    NestedStructuredSystem,
    build_structured_system,
    nested_prior_statistics,
)
from superglm.solvers._structured.nested import (
    NestedDataOperator,
    NestedPenalizedOperator,
    NestedStructuredLayout,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    compact_operator_diagonal,
)
from superglm.solvers._structured.state import (
    centred_data_operator,
)
from superglm.solvers.hessian_factor import HessianFactor
from superglm.solvers.irls_direct._finalise import _IRLSDirectFinalGeometry
from superglm.solvers.irls_direct._step import _IRLSDirectStep
from superglm.solvers.mode_score import centre_offset_mean, centred_data_score

if TYPE_CHECKING:
    from superglm.solvers._structured.selection import (
        StructuredBackendDecision,
    )
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.types import GroupSlice


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

    # Public as ``superglm.StructuredSolverError``: tracebacks and pickles name the
    # package that re-exports it, not this submodule.
    __module__ = "superglm.solvers.irls_direct"


@contextmanager
def _structured_solver_errors():
    """Raise a structured factor's ``np.linalg.LinAlgError`` as ``StructuredSolverError``.

    Wraps every structured factor operation of a fit.  The factors refuse
    with ``np.linalg.LinAlgError`` rather than truncate when they cannot
    certify a rank decision or represent a result.
    """
    try:
        yield
    except StructuredSolverError:
        raise
    except np.linalg.LinAlgError as error:
        # ``str`` leaves out the error's notes, so join them to the message
        notes = "".join(f" {str(note).rstrip('.')}." for note in getattr(error, "__notes__", ()))
        raise StructuredSolverError(
            f"The structured solver cannot proceed: {str(error).rstrip('.')}.{notes} This "
            "should not happen for a model its data identify; direct_solve='gram' fits it "
            "with the dense solver instead."
        ) from error


# Levenberg shifts for an observed iterate the factor refuses as not positive
# definite: increasing eps on the Jacobi-scaled Hessian, eps |H_ii| unscaled
# (Wood, Pya & Saefken 2016, section 3.1.2).  The sequence is fixed, so the
# shift is a function of the iterate alone and does not move the mode.
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

    A refused Fisher iterate propagates.  A refused observed iterate takes the
    smallest shift of ``_LEVENBERG_SHIFTS`` whose factor certifies; if none
    does, the unshifted refusal propagates with the largest shift's attached
    as a note.  The caller adds ``E beta`` to the right-hand side to make the
    shifted Newton step ``(H + E)^-1 g``.
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


def _structured_score_centre(
    system, factor, dm: DesignMatrix, W: NDArray, state_center: NDArray | None
) -> tuple:
    """``(mean_x, sum_w, centred diagonal, weakly identified slopes, mean_x - c)`` of a structured solve.

    The centring the certificate's score is formed about (``mode_residual``).
    The last entry is ``None`` without a centred state.
    """
    operator = system.operator
    xtw = np.empty(operator.shape[0], dtype=np.float64)
    xtw[operator.small_indices] = system.xtw_small
    xtw[operator.structured_indices] = system.xtw_structured
    mean_x = xtw / system.sum_w
    # an fs or sz system centres its c0-shifted moments, so a large-offset
    # column's centred diagonal does not cancel to noise
    diagonal = compact_operator_diagonal(centred_data_operator(system))
    excluded = tuple(
        index - 1 for index in getattr(factor, "weakly_identified_coefficients", ()) if index > 0
    )
    offset_mean = (
        None
        if state_center is None
        else centre_offset_mean(dm, W, float(system.sum_w), state_center, mean_x)
    )
    return mean_x, float(system.sum_w), diagonal, excluded, offset_mean


class _IRLSDirectStructuredStep(_IRLSDirectStep):
    """The fit's step on the structured backend: the random-effect chain, ``fs`` or ``sz`` term in block form.

    Chosen from the model's structure alone, never its data; its layout fixes
    the state's centre.  Keeps the dominant group, its layout and the last
    system solved, and no reference to the fit.  Its own finaliser publishes
    the profiled factor's ``log det H``, rank and EDF, with no dense
    ``RankInfo``.
    """

    __slots__ = ("_group_index", "_layout", "_last_system")
    backend = "structured"

    def __init__(
        self, dm: DesignMatrix, groups: list[GroupSlice], decision: StructuredBackendDecision
    ) -> None:
        group_index = decision.group_index
        if group_index is None:  # pragma: no cover - selection invariant
            raise RuntimeError("Structured backend has no dominant group.")
        layout = get_structured_layout(
            dm,
            groups,
            dominant_group_index=group_index,
            chain_group_indices=decision.chain_group_indices,
        )
        if isinstance(layout, FactorSmoothLeafLayout):
            # every dense kernel of the fs leaf route acts on its border; a
            # nested chain's parent blocks are not border-wide and keep the release
            keep_narrow_cap(len(layout.small_indices) + 1)
        self._group_index, self._layout = group_index, layout
        # the system the last step solved: the fREML performance iteration's export
        self._last_system: (
            FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem | None
        ) = None

    def state_centre(self, weights: NDArray, p: int) -> NDArray | None:
        """The PIRLS state's centre: the factor's prior-weighted border centre ``c0``, 0 on the tree.

        A nested chain's or ``fs`` term's state is ``(alpha, beta)`` about
        ``c0``, so eta is never formed by cancelling ``X beta`` against a raw
        intercept that absorbs a column's offset.  ``None`` for an ``sz`` term.
        """
        layout = self._layout
        if isinstance(layout, NestedStructuredLayout):
            border_center, _ = nested_prior_statistics(layout, weights)
        elif isinstance(layout, FactorSmoothLeafLayout):
            border_center, _ = factor_smooth_prior_statistics(layout, weights)
        else:
            return None
        state_center = np.zeros(p)
        state_center[layout.small_indices] = border_center
        return state_center

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        # the block system of [1 X]' W [1 X] + S and its right-hand side
        _t0 = time.perf_counter()
        z_off = fit.z - fit.offset
        W, dm, committed, newton_score = fit.W, fit.dm, fit.committed, fit.newton_score
        _state_center, recorder = fit._state_center, fit.recorder
        proposal_centred_intercept = fit.proposal_centred_intercept
        observed = fit.working_rows.curvature_source == "observed"
        Wz = W * z_off if newton_score is None else W * z_off + newton_score
        structured_system = build_structured_system(
            fit.gms,
            fit.groups,
            W,
            Wz,
            signed=observed,
            dominant_group_index=self._group_index,
            layout=self._layout,
            prior_weights=fit.weights,
        )
        penalized_operator = build_penalized_structured_operator(
            structured_system,
            fit.gms,
            fit.groups,
            fit.lambda2,
            reml_penalties=fit.reml_penalties,
            S_override=fit.S_override,
        )
        self._last_system = structured_system
        recorder.t_gram += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        with _structured_solver_errors():
            augmented_factor, rhs, levenberg_shift, shift_diagonal = _build_iterate_factor(
                structured_system, penalized_operator, observed=observed
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
        if levenberg_shift:
            recorder.count("irls_levenberg_shifts")
            recorder.peak("irls_levenberg_max_shift", levenberg_shift)
        beta = beta_aug[1:]
        if _state_center is not None:
            # the solve's intercept is the centred alpha about the same c0
            proposal_centred_intercept = float(beta_aug[0])
            intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
        else:
            intercept = float(beta_aug[0])
        if augmented_factor.rank_truncated and (
            isinstance(augmented_factor, SumToZeroTreeFactor) or newton_score is not None
        ):
            # On a truncated factor, step by the minimum-norm increment
            # ``Delta = H^+ g``, ``g = [1 X]' W (z - eta) - [0; S beta]`` (Pes &
            # Rodriguez 2021, arXiv 2101.07560, eqs. 1.2-1.4), not the
            # minimum-norm solution ``H^+ [1 X]'Wz``: the latter resets the
            # iterate along every truncated direction, which throws a level
            # with no finite mode (Geyer 2009, Theorem 4) arbitrarily far.  The
            # two agree in exact arithmetic when nothing is truncated.
            if newton_score is not None:
                # a mean-space Newton step taken as the increment on a
                # truncated structured factor
                recorder.count("irls_truncated_newton_increments")
            residual_rows = W * (fit.z - fit.eta) if newton_score is None else newton_score
            gradient = np.empty(fit.p + 1, dtype=np.float64)
            gradient[0] = float(np.sum(residual_rows))
            # The intercept entry is in the state's own coordinate (alpha about
            # the centre when centred), and a centred state's score is formed
            # on centred rows, ``(X - 1 c')' r``, so a large column offset never
            # cancels the border's ``X' r``.
            if _state_center is not None:
                gradient[1:] = centred_data_score(
                    dm, residual_rows, _state_center
                ) - fit.penalty_matvec(committed.beta)
                increment = augmented_factor.solve(
                    gradient,
                    centred=True,
                    border_centred=gradient[augmented_factor.small_indices[1:]],
                )
            else:
                gradient[1:] = dm.rmatvec(residual_rows) - fit.penalty_matvec(committed.beta)
                increment = augmented_factor.solve(gradient)
            beta = committed.beta + increment[1:]
            if _state_center is not None:
                assert committed.centred_intercept is not None
                proposal_centred_intercept = float(committed.centred_intercept) + float(
                    increment[0]
                )
                intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
            else:
                intercept = float(committed.intercept) + float(increment[0])
        fit._used_svd = False  # one path: the engine has no SVD fallback
        fit._cond_est = augmented_factor.schur_condition_estimate
        fit.rank_truncated = augmented_factor.rank_truncated
        if fit.convergence == "mode_score":
            fit._score_centre = _structured_score_centre(
                structured_system, augmented_factor, dm, W, _state_center
            )
        recorder.t_solve += time.perf_counter() - _t0
        fit.beta, fit.intercept = beta, intercept
        fit.proposal_centred_intercept = proposal_centred_intercept

    def finalise(self, fit: _IRLSDirectFit) -> _IRLSDirectFinalGeometry:
        """The retained model's block system and ``cache_out``, then its profiled factor.

        Moments stay in block form; the dominant ``K x K`` block is never
        materialised.  An fREML performance iteration publishes the system its
        last step solved, every other fit the export rows' system.
        """
        W, dm, p, gms, groups = fit.W, fit.dm, fit.p, fit.gms, fit.groups
        _return_working_system, cache_out = fit._return_working_system, fit.cache_out
        if _return_working_system:
            structured_final = self._last_system
            if structured_final is None:
                raise RuntimeError("working structured system was not computed")
        else:
            z_off = fit.z - fit.offset
            structured_final = build_structured_system(
                gms,
                groups,
                W,
                W * z_off,
                signed=fit.export_rows.curvature_source == "observed",
                dominant_group_index=self._group_index,
                layout=self._layout,
                prior_weights=fit.weights,
            )
        penalized_operator = build_penalized_structured_operator(
            structured_final,
            gms,
            groups,
            fit.lambda2,
            reml_penalties=fit.reml_penalties,
            S_override=fit.S_override,
        )
        XtWX_beta = structured_final.operator
        XtW1 = np.empty(p, dtype=np.float64)
        XtW1[XtWX_beta.small_indices] = structured_final.xtw_small
        XtW1[XtWX_beta.structured_indices] = structured_final.xtw_structured
        sum_W = structured_final.sum_w
        mean_x = XtW1 / sum_W

        # The cached-W fREML optimizer re-solves the profiled system for a new
        # penalty from these, with no data pass.
        if cache_out is not None:
            XtWz = np.empty(p, dtype=np.float64)
            XtWz[XtWX_beta.small_indices] = structured_final.xtwz_small
            XtWz[XtWX_beta.structured_indices] = structured_final.xtwz_structured
            sum_Wz = structured_final.sum_wz
            mean_z = sum_Wz / sum_W
            cache_out["structured_system"] = structured_final
            cache_out["structured_operator"] = XtWX_beta
            cache_out["penalized_operator"] = penalized_operator
            cache_out["xtwz_small"] = structured_final.xtwz_small
            cache_out["xtwz_structured"] = structured_final.xtwz_structured
            if not _return_working_system and isinstance(structured_final, FactorSmoothLeafSystem):
                # The retained estimability rule reads these (a nested chain
                # certifies its border on its own leaf-shifted moments).
                cache_out["structured_row_column_norm"] = cancelled_column_row_norms(
                    centred_data_operator(structured_final), dm, W
                )
            cache_out["XtWz"] = XtWz
            cache_out["XtW1"] = XtW1
            cache_out["sum_W"] = sum_W
            cache_out["sum_Wz"] = sum_Wz
            cache_out["centered_rhs"] = XtWz - XtW1 * mean_z
            cache_out["mean_x"] = mean_x
            cache_out["mean_z"] = mean_z

        # REML works in the full (intercept, slopes) coefficient space; the
        # profiled factor carries the intercept's Schur complement in block form.
        _t0 = time.perf_counter()
        with _structured_solver_errors():
            augmented_factor, _ = build_augmented_structured_factor(
                structured_final, penalized_operator
            )
        structured_factor = profile_structured_factor(
            structured_final, augmented_factor, sum_W, XtW1
        )
        if fit._compute_reml_geometry:
            XtWX_S_inv_beta: NDArray | HessianFactor = structured_factor
            with _structured_solver_errors():
                log_det_H: float | None = augmented_factor.logdet()
            reml_hessian_rank: int | None = augmented_factor.rank
        else:
            XtWX_S_inv_beta = np.empty((0, 0), dtype=np.float64)
            log_det_H = None
            reml_hessian_rank = None
        if fit._compute_fit_statistics:
            edf_operator = (
                CenteredBlockOperator(raw=XtWX_beta, cross=XtW1, total=sum_W, center=mean_x)
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
        fit.recorder.finalize_time(_t0)
        # Inference reads the structured factor directly; a dense RankInfo
        # would break the backend's O(K q + q^2) memory bound.
        return _IRLSDirectFinalGeometry(
            XtWX_beta=XtWX_beta,
            XtWX_S_inv_beta=XtWX_S_inv_beta,
            log_det_H=log_det_H,
            reml_hessian_rank=reml_hessian_rank,
            reml_geometry_summary=None,
            retained_reml_decomposition=None,
            p_eff=p_eff,
            rank_info=None,
            structured_factor=structured_factor,
            centered_final=None,
            XtW1=None,
        )
