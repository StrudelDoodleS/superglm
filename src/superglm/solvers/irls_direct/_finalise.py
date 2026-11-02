"""The direct IRLS fit's finalisers and what they return (``_IRLSDirectFinalGeometry``).

The structured step has its own finaliser; the QR, dense and QP steps share
``_finalise_centred``: the export rows' centred system, its slope
decomposition and ``RankInfo``, on a constrained fit's active face.
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

from superglm._roundoff import EPS as _EPS_FLOAT
from superglm._roundoff import gamma
from superglm.solvers.centered_system import (
    CenteredSystem,
    grouped_augmented_factor,
    grouped_projected_factor,
    grouped_structural_nulls,
    grouped_weighted_factor,
    penalty_factor,
)
from superglm.solvers.irls_direct._centred import _system_offset_mean
from superglm.solvers.pirls import REMLGeometrySummary
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankInfo,
    _formation_slack,
    decide_data_rank,
    decompose_factor,
    decompose_gram_if_authoritative,
    diagonal_of_square,
    face_pseudo_inverse,
    weights_fingerprint,
)

if TYPE_CHECKING:
    from superglm.group_matrix import DesignMatrix
    from superglm.solvers._structured.nested import (
        NestedDataOperator,
    )
    from superglm.solvers._structured.operators import (
        BlockSymmetricOperator,
        SumToZeroBlockOperator,
    )
    from superglm.solvers.hessian_factor import HessianFactor
    from superglm.solvers.irls_direct._fit import _CenteredFactorCertification, _IRLSDirectFit
    from superglm.solvers.irls_direct._step import _IRLSDirectStep
    from superglm.types import GroupSlice


@dataclass(frozen=True, slots=True, eq=False)
class _IRLSDirectFinalGeometry:
    """What a finaliser returns: the retained model's geometry, which the fit publishes.

    ``XtWX_beta`` is the data Gram ``return_xtwx`` returns (``X'WX`` centred, the block data
    operator structured); ``XtWX_S_inv_beta`` the profiled slope inverse (structured: the
    profiled factor); ``log_det_H`` and ``reml_hessian_rank`` the augmented Hessian's log
    determinant and rank; ``reml_geometry_summary`` and ``retained_reml_decomposition`` the
    centred O(p) REML summary and, when kept, its slope decomposition; ``p_eff`` the EDF (0
    without fit statistics); ``rank_info`` the centred data-subspace metadata;
    ``structured_factor`` the profiled factor the backend disclosure reads; ``centered_final``
    and ``XtW1`` the centred final system and raw ``X'W1``. REML fields are empty or ``None``
    without REML geometry, one route's fields ``None`` on the other; never the fit or a step.
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


def _active_face_basis(held: NDArray) -> NDArray | None:
    """An orthonormal basis ``Z`` of the coefficients the binding shape rows ``held`` leave free.

    The face is ``null(A_F)`` for the rows ``F`` of the QP's active set that
    bind at the fit (``_binding_rows``).  On it the constrained
    fit is, locally, the fit with those rows held as equalities, so its
    degrees of freedom and covariance are those of that fit (Meyer and
    Woodroofe 2000; Tibshirani and Taylor 2012, Theorem 3).  Columns no
    binding row touches keep their unit vectors; the rest of the null space
    is decided on unit rows at NumPy's ``matrix_rank`` cut.  ``None`` when no
    row binds.
    """
    if not held.shape[0]:
        return None
    width = held.shape[1]
    touched, vt, rank = _binding_rank(held)
    free = np.setdiff1d(np.arange(width), touched)
    basis = np.zeros((width, free.size + touched.size - rank))
    basis[free, np.arange(free.size)] = 1.0
    basis[np.ix_(touched, np.arange(free.size, basis.shape[1]))] = vt[rank:].T
    return basis


def _binding_rank(held: NDArray) -> tuple[NDArray, NDArray, int]:
    """The columns rows ``held`` touch, their unit rows' right singular vectors there, and their rank.

    The rank is NumPy's ``matrix_rank`` cut on the unit rows: the face's
    rank decision, which the editor shares.
    """
    touched = np.flatnonzero(np.any(held != 0.0, axis=0))
    unit = held[:, touched] / np.linalg.norm(held, axis=1)[:, None]
    _, singular, vt = np.linalg.svd(unit, full_matrices=True)
    return touched, vt, int(np.count_nonzero(singular > max(unit.shape) * _EPS_FLOAT * singular[0]))


def _on_face(matrix: NDArray, face: NDArray | None) -> NDArray:
    """``Z' M Z``, the matrix on the face ``Z``; ``M`` itself without a face."""
    return matrix if face is None else face.T @ matrix @ face


def _face_decomposition(
    matrix: NDArray, face: NDArray, factor: Callable[[], NDArray], rows: int
) -> RankDecomposition:
    """``Z' M Z`` decomposed as ``M`` is: its Gram when authoritative, else ``factor() Z``.

    ``M`` is PSD by construction (a nonnegative-weight Gram plus a penalty), so
    an indefinite ``Z' M Z`` is formation rounding, not curvature.  That, or a
    formation error reaching its smallest eigenvalue
    (``_face_formation_unresolved``), hands the decision to the factor.
    """
    on_face = _on_face(matrix, face)
    decomposition = decompose_gram_if_authoritative(on_face, psd_by_construction=True)
    if decomposition is not None and _face_formation_unresolved(matrix, face, on_face, rows):
        decomposition = None
    return decompose_factor(factor() @ face) if decomposition is None else decomposition


def _face_formation_unresolved(matrix: NDArray, face: NDArray, on_face: NDArray, rows: int) -> bool:
    """Whether forming ``Z' M Z`` may have moved it by as much as its smallest eigenvalue.

    Each entry of a Gram of ``rows`` rows is within ``gamma_rows`` of
    ``|X|'W|X|`` (Higham 2002, section 3.5), and ``(|X|'W|X|)_ij <=
    sqrt(M_ii M_jj)`` (Cauchy-Schwarz), so ``Z' M Z`` is within ``gamma_rows
    r_k r_l`` of its exact value, ``r = |Z|' sqrt(diag M)``:
    ``rank._formation_slack`` bounds that in ``Z' M Z``'s Jacobi scaling.  By
    Weyl's inequality a bound that reaches the smallest eigenvalue there
    leaves the exact face Gram's definiteness undecided.
    """
    diagonal = np.diag(on_face)
    active = diagonal > 0.0
    if not np.any(active):
        return False
    reach = (np.abs(face[:, active]).T @ np.sqrt(np.maximum(np.diag(matrix), 0.0))) ** 2
    slack = _formation_slack(gamma(rows) * reach, diagonal[active])
    scale = np.sqrt(diagonal[active])
    equilibrated = on_face[np.ix_(active, active)] / np.outer(scale, scale)
    return bool(slack >= np.linalg.eigvalsh(0.5 * (equilibrated + equilibrated.T))[0])


def _face_factor_edf(rows: NDArray, penalty: NDArray, face: NDArray) -> tuple[NDArray, NDArray]:
    """``diag(F)`` and ``diag(2F - F^2)``, ``F = Z M Z'``: the face fit's EDF and ``edf1``.

    ``rows`` is ``R`` with ``R'R = Z'DZ`` (``grouped_projected_factor``), so ``B = [R; P Z]``,
    ``P = penalty_factor(S)``, factors the face Hessian unformed and ``M = B^+ [R; 0]``: the
    influence trace from orthogonal factors (Wood 2011; Fasiolo et al. 2020, supplement B) and
    ``edf1`` of Wood (2013, sec. 2.4). Rows are projected before factoring, since Householder
    QR is backward stable only columnwise (Higham 2002, secs. 19.3-19.4) and a light face's
    curvature would be lost to heavy rows; each projected entry is within
    ``gamma_(p + T + 3) |a_r|' |Z|`` of exact. ``Z M Z'`` omits coupling to curvature off the
    face: zero trace per term, but entries up to the weight ratio, unresolvable in float64.
    """
    projected = np.vstack((rows, penalty_factor(penalty) @ face))
    decomposition = decompose_factor(projected, retain_factor_solve=True)
    data = np.zeros_like(projected)
    data[: rows.shape[0]] = rows
    influence = face @ decomposition.solve_factor_rhs(data) @ face.T
    edf = np.diag(influence).copy()
    return edf, 2.0 * edf - diagonal_of_square(influence)


def _feature_edf(
    centered: CenteredSystem,
    decomposition: RankDecomposition,
    factor: Callable[[int], NDArray],
    face: NDArray | None = None,
    inverse: NDArray | None = None,
) -> tuple[NDArray, NDArray | None]:
    """``diag((D + S)^+ D)``: each coefficient's effective degrees of freedom, and its ``edf1``.

    ``D`` is the centred data Gram, ``S`` the penalty; on an active face ``Z`` the diagonal is
    ``diag(Z (Z'(D + S)Z)^+ Z'D)``. Where the factor decided the face decomposition, the EDF and
    ``edf1`` are read off the projected rows ``factor(0)`` (``_face_factor_edf``); elsewhere
    ``edf1`` is ``None``. If the diagonal is not finite (subnormal weights overflow the
    pseudo-inverse), ``D`` and ``S`` are scaled by the power of two bringing the Hessian's
    largest entry into ``[1/2, 1)`` and refactored; PSD gives ``|S_ij| <= max diag(D + S)``, so
    the penalty cannot overflow. ``factor(exponent)`` gives the rows at scale ``2^-exponent``;
    ``inverse`` is ``face_pseudo_inverse(decomposition, face)`` when already formed.
    """
    if face is not None and decomposition.method == "qr_svd":
        return _face_factor_edf(factor(0), centered.penalty, face)
    if inverse is None:
        inverse = face_pseudo_inverse(decomposition, face)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        unscaled = np.diag(inverse @ centered.data_gram).copy()
    if np.all(np.isfinite(unscaled)):
        return unscaled, None
    hessian = centered.hessian
    largest = float(np.max(np.abs(hessian), initial=0.0))
    if largest == 0.0 or not math.isfinite(largest):
        return unscaled, None
    exponent = math.frexp(largest)[1]
    scaled = decompose_gram_if_authoritative(
        _on_face(np.ldexp(hessian, -exponent), face), psd_by_construction=True
    )
    if scaled is None and face is not None:
        return _face_factor_edf(factor(exponent), np.ldexp(centered.penalty, -exponent), face)
    if scaled is None:
        scaled = decompose_factor(factor(exponent))
    scaled_gram = np.ldexp(centered.data_gram, -exponent)
    return np.diag(face_pseudo_inverse(scaled, face) @ scaled_gram).copy(), None


def _retained_rank_info(
    centered_final: CenteredSystem,
    W: NDArray,
    augmented_rank: RankDecomposition,
    feature_edf: NDArray,
    feature_edf1: NDArray | None,
    *,
    dm: DesignMatrix,
    groups: list[GroupSlice],
    p: int,
    step_data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None,
    _return_working_system: bool,
    certify_centered_factor: Callable[..., _CenteredFactorCertification],
    face: NDArray | None,
    face_rows: NDArray | None,
    face_rank: RankDecomposition | None,
) -> RankInfo:
    """The data-subspace metadata only retained-fit inference reads.

    A pure function of the final centred system and weights, so it may run
    here or be deferred (``_defer_rank_info``) until the caller knows this fit
    is the one it publishes.  Owner and lifetime of a deferred call: the
    caller's model, until ``fit_ops._install_deferred_rank_info`` runs it or
    the next REML attempt drops it.  It reads ``dm`` when it runs, so nothing
    may mutate the design's group matrices in place in between.  Never a
    method of the fit, which it would keep alive (#492).
    """
    M_beta = centered_final.hessian + centered_final.sum_w * np.outer(
        centered_final.mean_x, centered_final.mean_x
    )

    def raw_factor() -> NDArray:
        return np.vstack(
            (
                certify_centered_factor(centered_final, W).factor,
                np.sqrt(centered_final.sum_w) * centered_final.mean_x,
            )
        )

    coefficient_rank = decompose_gram_if_authoritative(M_beta)
    if coefficient_rank is None:
        coefficient_rank = decompose_factor(raw_factor())
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
    if feature_edf1 is not None:
        feature_edf1.setflags(write=False)
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
        face=face,
        face_rows=face_rows,
        face_augmented=face_rank,
        face_coefficient=(
            None if face is None else _face_decomposition(M_beta, face, raw_factor, dm.n)
        ),
        feature_edf1=feature_edf1,
    )


def _final_centred_system(fit: _IRLSDirectFit) -> tuple[CenteredSystem, NDArray, NDArray]:
    """The retained model's centred system, its raw ``X'WX`` and ``X'W1``, and ``cache_out``.

    An fREML performance iteration (``_return_working_system``) takes the
    system its last step solved.
    """
    _return_working_system, W, cache_out = fit._return_working_system, fit.W, fit.cache_out
    dm = fit.dm
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

    # The cached-W fREML optimizer re-solves the profiled system for a new
    # penalty from these, with no data pass (O(p^3), not O(n K^2)).
    if cache_out is not None:
        cache_out["XtWX"] = XtWX
        cache_out["centered_XtWX"] = centered_final.data_gram
        # the pair the Gram was centred about (``(mean_x, None)`` without a dense column)
        cache_out["centre_pair"] = centered_final.centre_pair()
        # mean_x less the state's centre, for a cached trial's centred intercept
        cache_out["centre_offset_mean"] = offset_mean_final
        # the matrix the slope decomposition was taken of (``reml.identified``)
        cache_out["centered_hessian"] = centered_final.hessian
        # the rows its Gram summed, the length of its formation bound
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
    """The finaliser the QR, dense and QP steps share.

    The retained centred system, its profiled slope decomposition, ``log det
    H``, the REML summary, the EDF and ``RankInfo`` (computed here or
    deferred).  A constrained fit's statistics are its active face's.  A
    step's own data rank is computed here, never deferred, so no deferred
    callable keeps the step.
    """
    centered_final, XtWX, XtW1 = _final_centred_system(fit)
    W, dm = fit.W, fit.dm
    _defer_rank_info = fit._defer_rank_info
    # Profiling the intercept gives the Schur complement H_c, whose inverse is
    # the slope block of H_aug^-1 and log|H_aug| = log(sum W) + log|H_c|.  With
    # aliases this is the centred-space measure, not the raw pseudo-determinant.
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
        # A constrained fit's statistics are its active face's.
        qp = fit.qp
        face_rows = (
            None
            if qp is None
            else qp.binding_rows(
                fit.beta, np.sqrt(np.maximum(np.diag(centered_final.data_gram), 0.0))
            )
        )
        face = None if face_rows is None else _active_face_basis(face_rows)

        def augmented_factor(exponent: int = 0) -> NDArray:
            # unscaled, the certified factor the face decomposition reads
            if not exponent:
                return fit.certify_centered_factor(final_system, W).factor
            return grouped_augmented_factor(
                dm,
                np.ldexp(W, -exponent),
                np.ldexp(final_system.penalty, -exponent),
                center=final_system.mean_x,
            )

        def projected_rows(exponent: int = 0) -> NDArray:
            # the data rows on the face, the rows the face's EDF is read off
            centre, centre_lo = final_system.centre_pair()
            return grouped_projected_factor(
                dm, np.ldexp(W, -exponent), face, center=centre, center_lo=centre_lo
            )[0]

        face_rank = (
            None
            if face is None
            else _face_decomposition(centered_final.hessian, face, augmented_factor, dm.n)
        )
        feature_edf, feature_edf1 = _feature_edf(
            centered_final,
            augmented_rank if face_rank is None else face_rank,
            augmented_factor if face is None else projected_rows,
            face,
            # off a face, the inverse the REML geometry formed
            inverse=XtWX_S_inv_beta if face is None else None,
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
            feature_edf1,
            dm=dm,
            groups=fit.groups,
            p=fit.p,
            step_data_rank=step_data_rank,
            _return_working_system=fit._return_working_system,
            certify_centered_factor=fit.certify_centered_factor,
            face=face,
            face_rows=face_rows,
            face_rank=face_rank,
        )

        if fit.compute_rank_info or (_defer_rank_info is not None and step_data_rank is not None):
            # Deferring a step's own data rank would keep the step (and the QR
            # step's dense design) alive until install.
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
