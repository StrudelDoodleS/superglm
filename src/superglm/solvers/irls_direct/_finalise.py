"""The direct IRLS fit's finalisers: what each returns, and the centred one four steps name.

After the loop and its shared blocks (``_IRLSDirectFit._finish``, the step's
``terminal`` and the export rows), the fit runs the finaliser its step names
(``_IRLSDirectStep.finalise``) and publishes what it returns, an
``_IRLSDirectFinalGeometry``.  There are two.  The structured step's is its own
(``_IRLSDirectStructuredStep.finalise``): the block-form moments and the
profiled factor.  The QR, dense, QP and SCOP steps name the centred one here
(``_finalise_centred``): the centred system of the export rows, its slope
decomposition and ``RankInfo``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.solvers.centered_system import (
    CenteredSystem,
    grouped_augmented_factor,
    grouped_projected_factor,
    grouped_structural_nulls,
    grouped_weighted_factor,
)
from superglm.solvers.irls_direct._centred import _system_offset_mean
from superglm.solvers.pirls import REMLGeometrySummary
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankInfo,
    decide_data_rank,
    decompose_factor,
    decompose_gram_if_authoritative,
    weights_fingerprint,
)

if TYPE_CHECKING:
    from superglm.group_matrix import DesignMatrix
    from superglm.solvers.hessian_factor import HessianFactor
    from superglm.solvers.irls_direct._fit import _CenteredFactorCertification, _IRLSDirectFit
    from superglm.solvers.irls_direct._step import _IRLSDirectStep
    from superglm.solvers.structured import (
        BlockSymmetricOperator,
        NestedDataOperator,
        SumToZeroBlockOperator,
    )
    from superglm.types import GroupSlice


@dataclass(frozen=True, slots=True, eq=False)
class _IRLSDirectFinalGeometry:
    """What a finaliser returns: the retained model's geometry, which the fit publishes.

    - ``XtWX_beta``: the data Gram ``return_xtwx`` returns, ``X'WX`` on the
      centred route and the block data operator on the structured one;
    - ``XtWX_S_inv_beta``: the profiled slope inverse, the structured route's
      profiled factor; empty without REML geometry;
    - ``log_det_H`` and ``reml_hessian_rank``: the augmented Hessian's log
      determinant and rank, ``None`` without REML geometry;
    - ``reml_geometry_summary`` and ``retained_reml_decomposition``: the
      centred route's O(p) summary for the REML loop and, when the caller keeps
      it, its slope decomposition;
    - ``p_eff``: the effective degrees of freedom, 0 without fit statistics;
    - ``rank_info``: the centred route's data-subspace metadata, ``None`` when
      it is not asked for or is deferred;
    - ``structured_factor``: the structured route's profiled factor, which the
      backend disclosure reads;
    - ``centered_final`` and ``XtW1``: the centred route's final system and its
      raw ``X'W1``, which the SCOP step's post-fit inference reads.

    A field of one route is ``None`` on the other.  It holds arrays and
    factors, never the fit or a step.
    """

    XtWX_beta: NDArray | BlockSymmetricOperator | SumToZeroBlockOperator | NestedDataOperator
    XtWX_S_inv_beta: NDArray | HessianFactor
    log_det_H: float | None  # noqa: N815
    reml_hessian_rank: int | None
    reml_geometry_summary: REMLGeometrySummary | None
    retained_reml_decomposition: RankDecomposition | None
    p_eff: float
    rank_info: RankInfo | None
    structured_factor: HessianFactor | None
    centered_final: CenteredSystem | None
    XtW1: NDArray | None


def _feature_edf(
    centered: CenteredSystem,
    decomposition: RankDecomposition,
    factor: Callable[[int], NDArray],
) -> NDArray:
    """``diag((D + S)^+ D)``: each coefficient's effective degrees of freedom.

    ``D`` is the centred data Gram and ``S`` the penalty, ``D + S`` the
    Hessian ``decomposition`` factors.  The trace is invariant under scaling
    ``D`` and ``S`` jointly by one constant, but the pseudo-inverse is not: at
    a data scale ``s`` its entries are of order ``1 / s``, which overflows
    once ``s`` falls below ``1 / max float``, about 5.6e-309 (subnormal
    weights give ``s ~ 1e-317``).  The unscaled trace is formed first and,
    when every entry is finite, returned as it is: the unscaled computation,
    bit for bit, with no scan of the Hessian.  Otherwise the pseudo-inverse
    overflowed, and ``D`` and ``S`` are multiplied by the one power of two
    that brings the Hessian's largest entry into ``[1/2, 1)``, exact for
    normal and subnormal entries alike, and the scaled Hessian is factored
    again (``factor(exponent)`` when its Gram is not authoritative).  Both
    are positive semidefinite, so ``|S_ij| <= sqrt(S_ii S_jj) <= max diag(D
    + S)``: the scaled penalty is at most 1 and cannot overflow; a penalty
    that dominates the data leaves ``D``'s scaled entries small and the
    trace near its limit, 0.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        unscaled = np.diag(decomposition.pseudo_inverse() @ centered.data_gram).copy()
    if np.all(np.isfinite(unscaled)):
        return unscaled
    hessian = centered.hessian
    largest = float(np.max(np.abs(hessian), initial=0.0))
    if largest == 0.0 or not math.isfinite(largest):
        return unscaled
    exponent = math.frexp(largest)[1]
    scaled = decompose_gram_if_authoritative(np.ldexp(hessian, -exponent))
    if scaled is None:
        scaled = decompose_factor(factor(exponent))
    return np.diag(scaled.pseudo_inverse() @ np.ldexp(centered.data_gram, -exponent)).copy()


def _retained_rank_info(
    centered_final: CenteredSystem,
    W: NDArray,
    augmented_rank: RankDecomposition,
    feature_edf: NDArray,
    *,
    dm: DesignMatrix,
    groups: list[GroupSlice],
    p: int,
    step_data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None,
    _return_working_system: bool,
    certify_centered_factor: Callable[..., _CenteredFactorCertification],
) -> RankInfo:
    """The data-subspace metadata only retained-fit inference reads.

    A pure function of this fit's final centred system and
    weights, so it is the same whether it runs here or, for a
    caller that does not yet know whether this fit is the one it
    publishes (``_defer_rank_info``), after the solve returns.

    Owner and lifetime: the caller's model, from the solve until
    ``fit_ops._install_deferred_rank_info`` runs it or the next
    REML attempt drops it. The centred system, weights, augmented
    decomposition and EDF are bound by ``partial`` with the design
    (``dm``, read by the data-rank factor route), which is read when it
    runs, so the invariant is that nothing mutates the design's group
    matrices in place in between (``update_reml_r_inv`` writes the
    specs; ``release_leaf_memo`` drops a memo only). A step's own data
    rank (``step_data_rank``, the QR step's on its dense design) is never
    deferred (``_final_geometry``), so no step is kept.  It is never a
    method of the fit object, which it would keep alive (#492).
    """
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
    if step_data_rank is not None:
        data_rank = step_data_rank(centered_final, W)
    else:
        # The export rows are Fisher (``prefer_observed=False``), whose
        # weights are nonnegative: the data Gram is PSD by construction,
        # so a negative eigenvalue at its null is formation rounding and
        # sends the rank to the factor below instead of raising.
        final_centre, final_centre_lo = centered_final.centre_pair()

        def data_factor_rank() -> RankDecomposition:
            if not np.any(centered_final.penalty):
                return certify_centered_factor(centered_final, W).decomposition
            return decompose_factor(
                grouped_weighted_factor(
                    dm,
                    W,
                    center=final_centre,
                    center_lo=final_centre_lo,
                )
            )

        # Unpenalized, the data Gram is the Hessian, whose factor
        # certificate the fit already holds for its standard
        # errors (``reml_slope_rank``); reading it costs nothing.
        data_rank = decide_data_rank(
            centered_final.data_gram,
            data_factor_rank,
            project=lambda basis: grouped_projected_factor(
                dm, W, basis, center=final_centre, center_lo=final_centre_lo
            ),
            structural=lambda: grouped_structural_nulls(
                dm, W, center=final_centre, center_lo=final_centre_lo
            ),
            rows=dm.n,
            psd_by_construction=not _return_working_system,
            certify=not np.any(centered_final.penalty),
        )
    group_edf = {g.name: float(np.sum(feature_edf[g.sl])) for g in groups}
    selected_columns = np.arange(p, dtype=int)
    selected_columns.setflags(write=False)
    feature_edf.setflags(write=False)
    return RankInfo(
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
        data_gram=centered_final.data_gram,
        data_gram_weights=weights_fingerprint(W),
    )


def _final_centred_system(fit: _IRLSDirectFit) -> tuple[CenteredSystem, NDArray, NDArray]:
    """F08 and F09: the retained model's centred system, its raw ``X'WX`` and ``X'W1``, and ``cache_out``.

    The export rows' system through the fit's centred system and its caches;
    an fREML performance iteration (``_return_working_system``) takes the
    system its last step solved.
    """
    _return_working_system, W, cache_out = fit._return_working_system, fit.W, fit.cache_out
    dm = fit.dm
    # Preserve the raw coefficient-space payload used by REML separately from
    # the centered inference system.
    if _return_working_system:
        if fit._last_working_centered is None:
            raise RuntimeError("working centered system was not computed")
        centered_final = fit._last_working_centered
        offset_mean_final = fit._last_working_offset_mean
    else:
        z_off = fit.z - fit.offset
        centered_final = fit.get_centered_system(W, z_off)
        offset_mean_final = (
            None
            if fit._state_center is None or cache_out is None
            else _system_offset_mean(dm, W, centered_final, fit._state_center)
        )
    XtWX, XtW1, XtWz, sum_Wz = centered_final.raw_weighted_moments()
    sum_W = centered_final.sum_w

    # Cache final-iteration raw and stable centered quantities for the cached-W
    # fREML optimizer. These allow re-solving the profiled-intercept system with
    # a new penalty matrix S without any data passes (O(p³), not O(n·K²)).
    if cache_out is not None:
        cache_out["XtWX"] = XtWX
        cache_out["centered_XtWX"] = centered_final.data_gram
        # the pair the Gram was centred about (``(mean_x, None)`` without a dense column)
        cache_out["centre_pair"] = centered_final.centre_pair()
        # mean_x less the state's centre, formed on centred rows: a cached
        # trial's centred intercept (``reml.discrete``) reads it
        cache_out["centre_offset_mean"] = offset_mean_final
        # the matrix the slope decomposition was taken of (the identified
        # part of the Laplace approximation restricts it, ``reml.identified``)
        cache_out["centered_hessian"] = centered_final.hessian
        # the rows its Gram summed: its formation bound's length
        # (``reml.discrete._profiled_formation_error``)
        cache_out["n_rows"] = int(dm.n)
        cache_out["XtWz"] = XtWz
        cache_out["XtW1"] = XtW1
        cache_out["sum_W"] = sum_W
        cache_out["sum_Wz"] = sum_Wz
        cache_out["centered_rhs"] = centered_final.rhs
        cache_out["mean_x"] = centered_final.mean_x
        cache_out["mean_z"] = centered_final.mean_z
    return centered_final, XtWX, XtW1


def _finalise_centred(step: _IRLSDirectStep, fit: _IRLSDirectFit) -> _IRLSDirectFinalGeometry:
    """The centred finaliser, the ``finalise`` the QR, dense, QP and SCOP steps name (F08 to F11).

    The retained model's centred system (``_final_centred_system``), then its
    profiled slope decomposition, ``log det H``, the REML summary, the EDF
    and ``RankInfo``, computed here or deferred (``_retained_rank_info``).  A
    step's own data rank (``data_rank``, the QR step's on its dense design) is
    computed here, never deferred, so no deferred callable keeps the step.
    """
    centered_final, XtWX, XtW1 = _final_centred_system(fit)
    W, dm = fit.W, fit.dm
    _defer_rank_info = fit._defer_rank_info
    # REML works in the full (intercept, slopes) coefficient space.  Profiling
    # the unpenalized intercept yields the centered Schur complement H_c.  Its
    # inverse is the slope block of H_aug^{-1}, while the unit-determinant
    # centering transform gives log|H_aug| = log(sum(W)) + log|H_c| at full
    # rank. With aliases, the same expression is the retained centered-space
    # determinant measure, not the raw augmented pseudo-determinant.
    _t0 = time.perf_counter()
    reml_geometry_summary: REMLGeometrySummary | None = None
    XtWX_beta = XtWX
    reml_slope_rank: RankDecomposition | None
    if fit._compute_reml_geometry:
        reml_slope_rank = fit.authoritative_gram_rank(centered_final.hessian)
        if reml_slope_rank is None:
            certification = fit.certify_centered_factor(
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
            mean_hi=centered_final.mean_hi,
            mean_lo=centered_final.mean_lo,
        )
    else:
        reml_slope_rank = None
        XtWX_S_inv_beta = np.empty((0, 0), dtype=np.float64)
        log_det_H = None
        reml_hessian_rank = None
    retained_reml_decomposition = reml_slope_rank if fit._retain_reml_decomposition else None

    if fit._compute_fit_statistics:
        if reml_slope_rank is None:  # pragma: no cover - validated above
            raise RuntimeError("fit statistics require generic REML geometry")
        augmented_rank = reml_slope_rank
        final_system = centered_final
        feature_edf = _feature_edf(
            centered_final,
            augmented_rank,
            lambda exponent: grouped_augmented_factor(
                dm,
                np.ldexp(W, -exponent),
                np.ldexp(final_system.penalty, -exponent),
                center=final_system.mean_x,
            ),
        )
        feature_edf[np.abs(feature_edf) < 100.0 * np.finfo(float).eps] = 0.0
        p_eff = 1.0 + float(np.sum(feature_edf))

        step_data_rank = step.data_rank
        retained_rank_info = partial(
            _retained_rank_info,
            centered_final,
            W,
            augmented_rank,
            feature_edf,
            dm=dm,
            groups=fit.groups,
            p=fit.p,
            step_data_rank=step_data_rank,
            _return_working_system=fit._return_working_system,
            certify_centered_factor=fit.certify_centered_factor,
        )

        if fit.compute_rank_info or (_defer_rank_info is not None and step_data_rank is not None):
            # A deferred step's own data rank would keep the step, and the
            # QR step's dense (n, p) design, alive until install; it is
            # computed here.
            rank_info = retained_rank_info()
        else:
            rank_info = None
            if _defer_rank_info is not None:
                _defer_rank_info.append(retained_rank_info)
    else:
        p_eff = 0.0
        rank_info = None
    fit.recorder.finalize_time(_t0)
    return _IRLSDirectFinalGeometry(
        XtWX_beta=XtWX_beta,
        XtWX_S_inv_beta=XtWX_S_inv_beta,
        log_det_H=log_det_H,
        reml_hessian_rank=reml_hessian_rank,
        reml_geometry_summary=reml_geometry_summary,
        retained_reml_decomposition=retained_reml_decomposition,
        p_eff=p_eff,
        rank_info=rank_info,
        structured_factor=None,
        centered_final=centered_final,
        XtW1=XtW1,
    )
