"""Authoritative latent-coordinate geometry for shape-constrained REML."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_centered import (
    _raw_centering_well_scaled,
    centered_rhs,
)
from superglm.distributions import _VARIANCE_FLOOR, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import stabilize_eta
from superglm.reml.observed_geometry import compute_scop_observed_information_weights
from superglm.solvers.centered_system import (
    build_anchor_centered_system,
    build_centered_system,
    grouped_weighted_factor,
    penalty_factor,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import (
    RankDecomposition,
    decompose_factor,
    decompose_gram,
    diagonal_of_square,
    needs_factor_certification,
)
from superglm.types import GroupSlice


def _readonly(values: NDArray) -> NDArray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _dot_product_roundoff_factor(length: int) -> float:
    """Return Higham's ``gamma_k`` bound for a length-``k`` dot product."""
    scaled_epsilon = float(length) * np.finfo(np.float64).eps
    if scaled_epsilon >= 1.0:  # pragma: no cover - impossible for resident arrays
        return float("inf")
    return scaled_epsilon / (1.0 - scaled_epsilon)


SCOPCurvatureSource = Literal["observed", "fisher", "mixed"]


@dataclass(frozen=True)
class SCOPJointGeometry:
    """One coherent latent Hessian/decomposition for SCOP LAML consumers."""

    centered_hessian: NDArray
    hessian_inverse: NDArray
    transformed_intercept_cross: NDArray
    sum_w: float
    log_det_H: float  # noqa: N815
    hessian_rank: int
    curvature_source: SCOPCurvatureSource
    transformed_mean_x: NDArray | None = None


@dataclass(frozen=True)
class SCOPInferenceInfo:
    """Pya--Wood post-fit inference in public mapped-coefficient coordinates."""

    coefficient_inverse: NDArray
    augmented_inverse: NDArray
    intercept_edf: float
    feature_edf: NDArray
    feature_edf1: NDArray
    group_edf: Mapping[str, float]
    curvature_source: SCOPCurvatureSource

    @property
    def total_edf(self) -> float:
        return self.intercept_edf + float(np.sum(self.feature_edf))


@dataclass(frozen=True)
class SCOPModeScore:
    """Penalized likelihood score in joint ordinary/latent coordinates."""

    intercept: float
    slopes: NDArray
    max_abs: float
    relative_max: float


def _jacobian_diag(state: dict) -> NDArray:
    reparam = state.get("reparam")
    if reparam is not None and hasattr(reparam, "jacobian_diagonal"):
        beta_eff = np.asarray(state["beta_eff"], dtype=np.float64)
        if beta_eff.ndim != 1 or not np.all(np.isfinite(beta_eff)):
            raise ValueError("SCOP beta_eff must be a finite vector")
        return np.asarray(reparam.jacobian_diagonal(beta_eff), dtype=np.float64)
    gamma_eff = state.get("gamma_eff")
    if gamma_eff is not None:
        gamma_eff = np.asarray(gamma_eff, dtype=np.float64)
        if gamma_eff.ndim == 1 and np.all(np.isfinite(gamma_eff)):
            return gamma_eff
    beta_eff = np.asarray(state["beta_eff"], dtype=np.float64)
    if beta_eff.ndim != 1 or not np.all(np.isfinite(beta_eff)):
        raise ValueError("SCOP beta_eff must be a finite vector")
    return np.exp(np.clip(beta_eff, -500.0, 500.0))


def _second_derivative_diag(state: dict) -> NDArray:
    """Return elementwise second derivatives of the latent coefficient map."""
    reparam = state.get("reparam")
    if reparam is not None and hasattr(reparam, "second_derivative_diagonal"):
        beta_eff = np.asarray(state["beta_eff"], dtype=np.float64)
        return np.asarray(
            reparam.second_derivative_diagonal(beta_eff),
            dtype=np.float64,
        )
    return _jacobian_diag(state)


def _joint_jacobian_diag(width: int, scop_states: Mapping[int, dict]) -> NDArray:
    """Return d(mapped coefficients)/d(latent coefficients) for all slopes."""
    result: NDArray = np.ones(width, dtype=np.float64)
    claimed: NDArray = np.zeros(width, dtype=bool)
    for state in scop_states.values():
        group_slice = state["group_sl"]
        indices = np.arange(group_slice.start, group_slice.stop)
        jacobian = _jacobian_diag(state)
        if len(indices) != len(jacobian):
            raise ValueError("SCOP Jacobian width does not match its coefficient slice")
        if np.any(claimed[indices]):
            raise ValueError("SCOP coefficient slices must not overlap")
        claimed[indices] = True
        result[group_slice] = jacobian
    # Exponential shape-coordinate derivatives can underflow to zero. Every
    # mapping below multiplies by C and never forms C^{-1}, so retaining that
    # numerical boundary is both safe and more stable than rejecting it.
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("SCOP Jacobian entries must be non-negative and finite")
    return result


def scop_penalized_mode_score(
    *,
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    result: PIRLSResult,
    latent_penalty: NDArray,
    scop_states: Mapping[int, dict],
    centered_fisher_gram: NDArray,
    fisher_mean_x: NDArray,
    fisher_sum_w: float,
    eta_unclipped: NDArray | None = None,
) -> SCOPModeScore:
    """Certify the terminal Pya--Wood coefficient mode by its latent KKT score."""
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    offset_arr = np.asarray(offset_arr, dtype=np.float64)
    latent_penalty = np.asarray(latent_penalty, dtype=np.float64)
    centered_fisher_gram = np.asarray(centered_fisher_gram, dtype=np.float64)
    fisher_mean_x = np.asarray(fisher_mean_x, dtype=np.float64)
    if y.shape != (dm.n,) or sample_weight.shape != y.shape or offset_arr.shape != y.shape:
        raise ValueError("SCOP mode-score rows must match the design")
    if latent_penalty.shape != (dm.p, dm.p):
        raise ValueError("SCOP mode-score penalty must match the design")
    if centered_fisher_gram.shape != (dm.p, dm.p) or fisher_mean_x.shape != (dm.p,):
        raise ValueError("SCOP mode-score Fisher geometry must match the design")
    if not np.isfinite(fisher_sum_w) or fisher_sum_w <= 0.0:
        raise ValueError("SCOP mode-score Fisher weight sum must be positive and finite")

    beta_mapped = np.asarray(result.beta, dtype=np.float64)
    if beta_mapped.shape != (dm.p,):
        raise ValueError("SCOP mode-score coefficients must match the design")
    jacobian = _joint_jacobian_diag(dm.p, scop_states)
    beta_latent = beta_mapped.copy()
    for state in scop_states.values():
        group_slice = state["group_sl"]
        beta_eff = np.asarray(state["beta_eff"], dtype=np.float64)
        if beta_eff.shape != (group_slice.stop - group_slice.start,):
            raise ValueError("SCOP beta_eff must match its coefficient slice")
        beta_latent[group_slice] = beta_eff

    if eta_unclipped is None:
        eta_raw = dm.matvec(beta_mapped) + result.intercept + offset_arr
    else:
        eta_raw = np.asarray(eta_unclipped, dtype=np.float64)
        if eta_raw.shape != (dm.n,) or not np.all(np.isfinite(eta_raw)):
            raise ValueError("SCOP mode-score eta must be finite and match the design rows")
    eta = stabilize_eta(eta_raw, link)
    mu = clip_mu(link.inverse(eta), distribution)
    variance = np.maximum(
        np.asarray(distribution.variance(mu), dtype=np.float64),
        _VARIANCE_FLOOR,
    )
    derivative = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_score = sample_weight * (y - mu) * derivative / variance
    if not np.all(np.isfinite(row_score)):
        raise ValueError("SCOP penalized mode score is not finite")

    intercept_score = math.fsum(float(value) for value in row_score)
    mapped_scale = np.sqrt(np.maximum(np.diag(centered_fisher_gram), 0.0) / fisher_sum_w)
    raw_centering_safe = _raw_centering_well_scaled(fisher_mean_x, mapped_scale)
    if raw_centering_safe:
        centered_mapped_score = dm.rmatvec(row_score) - fisher_mean_x * intercept_score
    else:
        centered_mapped_score = centered_rhs(
            dm=dm,
            W=np.ones(dm.n, dtype=np.float64),
            mean_x=fisher_mean_x,
            z_centered=row_score,
        )
    data_score = jacobian * centered_mapped_score
    penalty_score = latent_penalty @ beta_latent
    slope_score = data_score - penalty_score

    max_abs = max(
        abs(intercept_score),
        float(np.max(np.abs(slope_score), initial=0.0)),
    )
    tiny = np.finfo(np.float64).tiny
    row_mass = max(tiny, float(np.sum(np.abs(row_score), dtype=np.float64)))
    latent_scale = mapped_scale * jacobian
    slope_scale = np.maximum(
        tiny,
        row_mass * latent_scale + np.abs(penalty_score),
    )
    # At a suppressed SCOP boundary, both the mapped data scale and the exact
    # penalty score can be zero.  A matrix-vector product through the
    # semidefinite penalty can still leave cancellation noise of order
    # eps * |S| |beta|; dividing that noise by ``tiny`` turns a numerical zero
    # into a unit relative KKT residual.  Classify only residuals covered by a
    # dimension-aware floating-point accumulation bound as zero.
    row_roundoff_factor = _dot_product_roundoff_factor(dm.n)
    intercept_ratio = abs(intercept_score) / row_mass
    if abs(intercept_score) <= row_roundoff_factor * row_mass:
        intercept_ratio = 0.0
    slope_ratio = np.abs(slope_score) / slope_scale
    data_roundoff = row_roundoff_factor * row_mass * latent_scale
    penalty_roundoff = _dot_product_roundoff_factor(dm.p) * (
        np.abs(latent_penalty) @ np.abs(beta_latent)
    )
    # A large semidefinite penalty can have a large multiplication-error bound
    # in an exact null direction.  That uncertainty must never erase an
    # independent data residual.  Classify a coordinate as numerical zero only
    # when each score contribution is itself within its own accumulation bound.
    roundoff_only = (np.abs(data_score) <= data_roundoff) & (
        np.abs(penalty_score) <= penalty_roundoff
    )
    slope_ratio[roundoff_only] = 0.0
    relative_max = max(
        intercept_ratio,
        float(np.max(slope_ratio, initial=0.0)),
    )
    return SCOPModeScore(
        intercept=float(intercept_score),
        slopes=_readonly(slope_score),
        max_abs=max_abs,
        relative_max=relative_max,
    )


def _augmented_inverse_from_mean(
    centered_inverse: NDArray,
    mean_x: NDArray,
    sum_w: float,
) -> NDArray:
    """Undo intercept profiling from a stable weighted design mean."""
    mean_x = np.asarray(mean_x, dtype=np.float64)
    inverse_mean = centered_inverse @ mean_x
    result = np.empty(
        (centered_inverse.shape[0] + 1, centered_inverse.shape[1] + 1),
        dtype=np.float64,
    )
    result[0, 0] = 1.0 / sum_w + float(mean_x @ inverse_mean)
    result[0, 1:] = -inverse_mean
    result[1:, 0] = -inverse_mean
    result[1:, 1:] = centered_inverse
    return result


type FisherWeights = NDArray | Callable[[], NDArray]


def _factor_certifier(
    *,
    dm: DesignMatrix | None,
    fisher_weights: FisherWeights | None,
    jacobian: NDArray,
    latent_penalty: NDArray,
) -> Callable[[NDArray | None], RankDecomposition] | None:
    """Return a lazy factor-space certifier for expected SCOP curvature."""
    if (dm is None) != (fisher_weights is None):
        raise ValueError("dm and fisher_weights must be provided together")
    if dm is None or fisher_weights is None:
        return None
    if dm.p != len(jacobian):
        raise ValueError("factor-certification design must match the slope dimension")

    resolved_weights: NDArray | None = None
    smooth_factor: NDArray | None = None

    def certify(center: NDArray | None) -> RankDecomposition:
        nonlocal resolved_weights, smooth_factor
        if resolved_weights is None:
            rows = fisher_weights() if callable(fisher_weights) else fisher_weights
            resolved_weights = np.asarray(rows, dtype=np.float64)
            if resolved_weights.shape != (dm.n,):
                raise ValueError("fisher_weights must match the design row count")
            if not np.all(np.isfinite(resolved_weights)) or np.any(resolved_weights < 0.0):
                raise ValueError("fisher_weights must be finite and non-negative")
        data_factor = grouped_weighted_factor(dm, resolved_weights, center=center)
        data_factor = data_factor * jacobian
        if smooth_factor is None:
            smooth_factor = penalty_factor(latent_penalty)
        factor = (
            data_factor if smooth_factor.shape[0] == 0 else np.vstack((data_factor, smooth_factor))
        )
        return decompose_factor(factor)

    return certify


def _scop_discarded_model_directions(
    scop_states: Mapping[int, dict],
    width: int,
) -> NDArray:
    """Lift the solver's discarded directions into model space, as rows.

    The joint solver truncates one factor spanning every SCOP group and hands
    each group its own columns of the result, so row ``r`` of every group's
    block belongs to the same joint direction and the groups reassemble
    row-wise. Groups solved one at a time instead carry directions confined to
    themselves, which is the shape this falls back to when the row counts do
    not agree.
    """
    blocks = []
    for state in scop_states.values():
        rows = state.get("discarded_directions")
        if rows is None:
            continue
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] == 0:
            continue
        blocks.append((state["group_sl"], rows))

    if not blocks:
        return np.zeros((0, width), dtype=np.float64)

    row_counts = {rows.shape[0] for _, rows in blocks}
    if len(row_counts) == 1 and len(blocks) > 1:
        nullity = row_counts.pop()
        directions = np.zeros((nullity, width), dtype=np.float64)
        for group_slice, rows in blocks:
            directions[:, group_slice] = rows
        return directions

    directions = np.zeros((sum(rows.shape[0] for _, rows in blocks), width), dtype=np.float64)
    offset = 0
    for group_slice, rows in blocks:
        directions[offset : offset + rows.shape[0], group_slice] = rows
        offset += rows.shape[0]
    return directions


def scop_resolved_range_projector(
    scop_states: Mapping[int, dict],
    width: int,
) -> NDArray | None:
    """Return the projector onto the range the SCOP steps could resolve.

    ``None`` when nothing was discarded, so callers leave their matrix
    untouched rather than multiplying by an identity.
    """
    discarded = _scop_discarded_model_directions(scop_states, width)
    if not discarded.size:
        return None
    frozen_basis, _ = np.linalg.qr(discarded.T)
    return np.eye(width) - frozen_basis @ frozen_basis.T


def restrict_to_scop_resolved_range(
    matrix: NDArray,
    scop_states: Mapping[int, dict],
) -> NDArray:
    """Project a joint latent matrix onto the resolved range, symmetrically."""
    projector = scop_resolved_range_projector(scop_states, matrix.shape[0])
    if projector is None:
        return matrix
    restricted = projector @ matrix @ projector
    return 0.5 * (restricted + restricted.T)


def scop_resolved_range_basis(
    scop_states: Mapping[int, dict],
    width: int,
) -> NDArray | None:
    """Orthonormal basis for the range the SCOP steps could resolve."""
    discarded = _scop_discarded_model_directions(scop_states, width)
    if not discarded.size:
        return None
    complete, _ = np.linalg.qr(discarded.T, mode="complete")
    return complete[:, discarded.shape[0] :]


def decompose_on_scop_resolved_range(
    matrix: NDArray,
    scop_states: Mapping[int, dict],
) -> RankDecomposition:
    """Decompose a joint latent matrix in resolved coordinates.

    Zeroing a frozen direction in place is not enough to make a determinant
    reproducible. ``decompose_gram`` equilibrates before it truncates, and
    equilibration rescales a direction whose diagonal has been driven to zero,
    lifting it back above the cutoff by an amount that depends on the rest of
    the matrix. Two assemblies of the same mode -- one from expected curvature,
    one from observed -- then disagree about a determinant that should be
    identical.

    Reducing to an explicit orthonormal basis removes the direction instead of
    scaling it, so the retained spectrum is the same object either way. The
    rank returned counts identified coordinates, which is what Wood's ``M_p``
    requires.
    """
    basis = scop_resolved_range_basis(scop_states, matrix.shape[0])
    if basis is None:
        return decompose_gram(matrix)
    reduced = basis.T @ matrix @ basis
    return decompose_gram(0.5 * (reduced + reduced.T))


def _decompose_with_factor_certification(
    matrix: NDArray,
    *,
    certifier: Callable[[NDArray | None], RankDecomposition] | None,
    center: NDArray | None,
) -> RankDecomposition:
    """Apply the shared Gram-first policy and consult rows only in its uncertainty band."""
    try:
        decomposition = decompose_gram(matrix)
    except ValueError:
        if certifier is None:
            raise
        return certifier(center)
    if certifier is not None and needs_factor_certification(decomposition):
        return certifier(center)
    return decomposition


def _decompose_on_certified_range(
    matrix: NDArray,
    certified: RankDecomposition,
) -> tuple[NDArray, NDArray, float, int]:
    """Decompose observed curvature on a factor-certified estimable range."""
    width = matrix.shape[0]
    null_basis = np.asarray(certified.parameter_null_basis, dtype=np.float64)
    nullity = width - certified.rank
    if null_basis.shape != (width, nullity) or nullity <= 0:
        raise ValueError("factor certification did not provide the expected null space")
    orthogonal, _ = np.linalg.qr(null_basis, mode="complete")
    estimable_basis = orthogonal[:, nullity:]
    reduced = estimable_basis.T @ matrix @ estimable_basis
    reduced = 0.5 * (reduced + reduced.T)
    decomposition = decompose_gram(reduced)
    if decomposition.rank != certified.rank:
        raise ValueError("observed curvature is singular inside the certified estimable range")
    projected = estimable_basis @ reduced @ estimable_basis.T
    inverse = estimable_basis @ decomposition.pseudo_inverse() @ estimable_basis.T
    return projected, inverse, decomposition.log_pdet, decomposition.rank


def build_scop_postfit_inference(
    *,
    raw_fisher_gram: NDArray,
    fisher_xtw: NDArray,
    fisher_sum_w: float,
    latent_penalty: NDArray,
    scop_states: Mapping[int, dict],
    groups: Sequence[GroupSlice],
    observed_geometry: SCOPJointGeometry,
    centered_fisher_gram: NDArray | None = None,
    fisher_mean_x: NDArray | None = None,
    dm: DesignMatrix | None = None,
    fisher_weights: FisherWeights | None = None,
) -> SCOPInferenceInfo:
    """Build Pya--Wood covariance and EDF from one terminal SCOP mode.

    Bayesian covariance uses the expected penalized Hessian (Pya--Wood
    supplementary material, S.5). Effective degrees of freedom instead use
    the retained full-Newton penalized Hessian multiplying expected data
    curvature (Pya--Wood Eq. 16). EDF allocation is evaluated in the
    intercept-profiled coordinate system.  That convention is invariant to
    design-column translation and avoids multiplying raw augmented moments
    whose large terms cancel only after matrix multiplication.

    ``centered_fisher_gram`` and ``fisher_mean_x`` should be supplied by the
    terminal centered system.  The raw-moment reconstruction remains only for
    synthetic/backward-compatible callers.  ``dm`` and ``fisher_weights``
    enable factor-space rank certification and are consulted lazily only when
    the shared Gram policy enters its uncertainty band.
    """
    raw_fisher_gram = np.asarray(raw_fisher_gram, dtype=np.float64)
    fisher_xtw = np.asarray(fisher_xtw, dtype=np.float64)
    latent_penalty = np.asarray(latent_penalty, dtype=np.float64)
    if raw_fisher_gram.ndim != 2 or raw_fisher_gram.shape[0] != raw_fisher_gram.shape[1]:
        raise ValueError("raw_fisher_gram must be square")
    width = raw_fisher_gram.shape[0]
    if fisher_xtw.shape != (width,):
        raise ValueError("fisher_xtw must match the slope dimension")
    if centered_fisher_gram is not None:
        centered_fisher_gram = np.asarray(centered_fisher_gram, dtype=np.float64)
        if centered_fisher_gram.shape != (width, width):
            raise ValueError("centered_fisher_gram must match the slope dimension")
    if fisher_mean_x is not None:
        fisher_mean_x = np.asarray(fisher_mean_x, dtype=np.float64)
        if fisher_mean_x.shape != (width,):
            raise ValueError("fisher_mean_x must match the slope dimension")
    if latent_penalty.shape != (width, width):
        raise ValueError("latent_penalty must match the slope dimension")
    if observed_geometry.centered_hessian.shape != (width, width):
        raise ValueError("observed SCOP geometry must match the slope dimension")
    if observed_geometry.hessian_inverse.shape != (width, width):
        raise ValueError("observed SCOP inverse must match the slope dimension")
    if observed_geometry.transformed_intercept_cross.shape != (width,):
        raise ValueError("observed SCOP intercept cross must match the slope dimension")
    if not np.isfinite(fisher_sum_w) or fisher_sum_w <= 0.0:
        raise ValueError("fisher_sum_w must be positive and finite")
    if not (
        np.all(np.isfinite(raw_fisher_gram))
        and np.all(np.isfinite(fisher_xtw))
        and np.all(np.isfinite(latent_penalty))
        and (centered_fisher_gram is None or np.all(np.isfinite(centered_fisher_gram)))
        and (fisher_mean_x is None or np.all(np.isfinite(fisher_mean_x)))
    ):
        raise ValueError("SCOP Fisher inference inputs must be finite")

    raw_fisher_gram = 0.5 * (raw_fisher_gram + raw_fisher_gram.T)
    latent_penalty = 0.5 * (latent_penalty + latent_penalty.T)
    if fisher_mean_x is None:
        fisher_mean_x = fisher_xtw / fisher_sum_w
    if centered_fisher_gram is None:
        centered_fisher_gram = raw_fisher_gram - (np.outer(fisher_xtw, fisher_xtw) / fisher_sum_w)
    centered_fisher_gram = 0.5 * (centered_fisher_gram + centered_fisher_gram.T)

    jacobian = _joint_jacobian_diag(width, scop_states)
    latent_data_gram = raw_fisher_gram * jacobian[:, None] * jacobian[None, :]
    latent_mean_x = fisher_mean_x * jacobian
    centered_data_gram = centered_fisher_gram * jacobian[:, None] * jacobian[None, :]
    certifier = _factor_certifier(
        dm=dm,
        fisher_weights=fisher_weights,
        jacobian=jacobian,
        latent_penalty=latent_penalty,
    )

    fisher_profile = _decompose_with_factor_certification(
        centered_data_gram + latent_penalty,
        certifier=certifier,
        center=fisher_mean_x,
    )
    fisher_centered_inverse = fisher_profile.pseudo_inverse()
    fisher_augmented_inverse = _augmented_inverse_from_mean(
        fisher_centered_inverse,
        latent_mean_x,
        fisher_sum_w,
    )
    augmented_jacobian = np.concatenate(([1.0], jacobian))
    mapped_augmented_inverse = (
        fisher_augmented_inverse * augmented_jacobian[:, None] * augmented_jacobian[None, :]
    )

    coefficient_decomposition = _decompose_with_factor_certification(
        latent_data_gram + latent_penalty,
        certifier=certifier,
        center=None,
    )
    coefficient_inverse = coefficient_decomposition.pseudo_inverse()
    mapped_coefficient_inverse = coefficient_inverse * jacobian[:, None] * jacobian[None, :]

    observed_mean_x = (
        np.asarray(observed_geometry.transformed_mean_x, dtype=np.float64)
        if observed_geometry.transformed_mean_x is not None
        else observed_geometry.transformed_intercept_cross / observed_geometry.sum_w
    )
    if observed_mean_x.shape != (width,) or not np.all(np.isfinite(observed_mean_x)):
        raise ValueError("observed SCOP mean must be a finite slope vector")
    delta_mean = latent_mean_x - observed_mean_x
    expected_data_at_observed_center = centered_data_gram + fisher_sum_w * np.outer(
        delta_mean,
        delta_mean,
    )
    # In coordinates [intercept at the observed weighted mean, slopes], the
    # observed penalized Hessian is block diagonal.  Keep the complete
    # off-diagonal expected-data blocks because they contribute to diag(F²).
    influence = np.empty((width + 1, width + 1), dtype=np.float64)
    influence[0, 0] = fisher_sum_w / observed_geometry.sum_w
    influence[0, 1:] = fisher_sum_w * delta_mean / observed_geometry.sum_w
    influence[1:, 0] = observed_geometry.hessian_inverse @ (fisher_sum_w * delta_mean)
    influence[1:, 1:] = observed_geometry.hessian_inverse @ expected_data_at_observed_center
    influence_edf = np.diag(influence).copy()
    influence_edf1 = 2.0 * influence_edf - diagonal_of_square(influence)
    near_zero = 100.0 * np.finfo(float).eps
    influence_edf[np.abs(influence_edf) < near_zero] = 0.0
    influence_edf1[np.abs(influence_edf1) < near_zero] = 0.0

    feature_edf = influence_edf[1:]
    feature_edf1 = influence_edf1[1:]
    group_edf = {group.name: float(np.sum(feature_edf[group.sl])) for group in groups}
    return SCOPInferenceInfo(
        coefficient_inverse=_readonly(mapped_coefficient_inverse),
        augmented_inverse=_readonly(mapped_augmented_inverse),
        intercept_edf=float(influence_edf[0]),
        feature_edf=_readonly(feature_edf),
        feature_edf1=_readonly(feature_edf1),
        group_edf=group_edf,
        curvature_source=observed_geometry.curvature_source,
    )


def build_cached_scop_joint_geometry(
    *,
    raw_fisher_gram: NDArray,
    fisher_xtw: NDArray,
    fisher_sum_w: float,
    latent_penalty: NDArray,
    scop_states: Mapping[int, dict],
    centered_fisher_gram: NDArray | None = None,
    fisher_mean_x: NDArray | None = None,
    dm: DesignMatrix | None = None,
    fisher_weights: FisherWeights | None = None,
) -> SCOPJointGeometry:
    """Build retained latent geometry from terminal SCOP Newton blocks.

    The ordinary and cross-term blocks use the final Fisher moments, while
    each SCOP diagonal block is replaced by the full-Newton block retained by
    the coefficient solver.  This is the exact Pya--Wood Eq. 16 geometry for
    canonical family/link combinations and records any per-block Fisher
    fallback used to keep the Newton solve positive definite.
    """
    raw_fisher_gram = np.asarray(raw_fisher_gram, dtype=np.float64)
    fisher_xtw = np.asarray(fisher_xtw, dtype=np.float64)
    latent_penalty = np.asarray(latent_penalty, dtype=np.float64)
    if raw_fisher_gram.ndim != 2 or raw_fisher_gram.shape[0] != raw_fisher_gram.shape[1]:
        raise ValueError("raw_fisher_gram must be square")
    width = raw_fisher_gram.shape[0]
    if fisher_xtw.shape != (width,):
        raise ValueError("fisher_xtw must match the slope dimension")
    if latent_penalty.shape != (width, width):
        raise ValueError("latent_penalty must match the slope dimension")
    if not np.isfinite(fisher_sum_w) or fisher_sum_w <= 0.0:
        raise ValueError("fisher_sum_w must be positive and finite")
    if (centered_fisher_gram is None) != (fisher_mean_x is None):
        raise ValueError("centered_fisher_gram and fisher_mean_x must be provided together")
    if fisher_mean_x is None:
        fisher_mean_x = fisher_xtw / fisher_sum_w
        centered_fisher_gram = raw_fisher_gram - (np.outer(fisher_xtw, fisher_xtw) / fisher_sum_w)
    else:
        fisher_mean_x = np.asarray(fisher_mean_x, dtype=np.float64)
        centered_fisher_gram = np.asarray(centered_fisher_gram, dtype=np.float64)
        if fisher_mean_x.shape != (width,):
            raise ValueError("fisher_mean_x must match the slope dimension")
        if centered_fisher_gram.shape != (width, width):
            raise ValueError("centered_fisher_gram must match the slope dimension")
    if not (
        np.all(np.isfinite(raw_fisher_gram))
        and np.all(np.isfinite(fisher_xtw))
        and np.all(np.isfinite(centered_fisher_gram))
        and np.all(np.isfinite(fisher_mean_x))
        and np.all(np.isfinite(latent_penalty))
    ):
        raise ValueError("cached SCOP geometry inputs must be finite")

    jacobian = _joint_jacobian_diag(width, scop_states)
    transformed_mean = fisher_mean_x * jacobian
    transformed_cross = fisher_sum_w * transformed_mean
    expected_centered = (
        centered_fisher_gram * jacobian[:, None] * jacobian[None, :] + latent_penalty
    )
    centered = expected_centered.copy()
    fallback_flags: list[bool] = []
    for state in scop_states.values():
        fallback = bool(state.get("last_fisher_fallback", False))
        fallback_flags.append(fallback)
        if fallback:
            # The solver may have added a diagonal ridge solely to obtain a
            # descent direction.  Wood's determinant and EDF use the exact
            # expected data curvature plus the model penalty, never that ridge.
            continue
        group_slice = state["group_sl"]
        retained_block = state.get("H_scop_penalized")
        if retained_block is None:
            raise RuntimeError("terminal SCOP Newton curvature was not retained")
        retained_block = np.asarray(retained_block, dtype=np.float64)
        block_width = group_slice.stop - group_slice.start
        if retained_block.shape != (block_width, block_width):
            raise ValueError("retained SCOP Newton block has the wrong shape")
        block_cross = transformed_cross[group_slice]
        centered[group_slice, group_slice] = retained_block - (
            np.outer(block_cross, block_cross) / fisher_sum_w
        )

    # This builder assembles the joint curvature directly from the expected
    # gram rather than through ``assemble_joint_hessian``, so it has to apply
    # the same restriction: the retained blocks arrive resolved, but the
    # expected cross-terms around them still reach the directions the SCOP
    # steps froze. Leaving that leakage here would give this determinant a
    # different value from the one ``assemble_joint_hessian`` produces for the
    # same mode, and the two are compared.
    centered = restrict_to_scop_resolved_range(centered, scop_states)
    expected_centered = restrict_to_scop_resolved_range(expected_centered, scop_states)

    if fallback_flags and all(fallback_flags):
        curvature_source: SCOPCurvatureSource = "fisher"
    elif any(fallback_flags):
        curvature_source = "mixed"
    else:
        curvature_source = "observed"
    certifier = _factor_certifier(
        dm=dm,
        fisher_weights=fisher_weights,
        jacobian=jacobian,
        latent_penalty=latent_penalty,
    )
    restricted_decomposition: tuple[NDArray, NDArray, float, int] | None = None
    try:
        decomposition = _decompose_with_factor_certification(
            centered,
            certifier=certifier if curvature_source == "fisher" else None,
            center=fisher_mean_x,
        )
        if (
            curvature_source != "fisher"
            and certifier is not None
            and needs_factor_certification(decomposition)
        ):
            certified = certifier(fisher_mean_x)
            # Near a suppressed shape boundary, roundoff can lift the exact
            # penalty null direction just enough for Cholesky to report a
            # discontinuous full rank. Equal integer ranks can also retain
            # different cutoff-boundary subspaces. Use the row/penalty factor
            # to certify the estimable range, then retain the observed
            # curvature and determinant on that range.
            if certified.rank < certified.width:
                restricted_decomposition = _decompose_on_certified_range(
                    centered,
                    certified,
                )
            elif certified.rank != decomposition.rank:
                # Fisher certifies only estimability, not the observed
                # curvature values. If it certifies the full space while the
                # observed Gram truncates one, let the established fallback
                # below replace the entire geometry with Fisher curvature.
                raise ValueError("observed curvature is singular on the certified full space")
    except ValueError:
        centered = expected_centered
        decomposition = _decompose_with_factor_certification(
            centered,
            certifier=certifier,
            center=fisher_mean_x,
        )
        curvature_source = "fisher"
        restricted_decomposition = None
    if restricted_decomposition is None:
        hessian_inverse = decomposition.pseudo_inverse()
        log_pdet = decomposition.log_pdet
        hessian_rank = decomposition.rank
    else:
        centered, hessian_inverse, log_pdet, hessian_rank = restricted_decomposition

    # The determinant and the identified-coordinate count come from the
    # resolved range, exactly as ``reml_laml_objective`` takes them, so a mode
    # reproduced from stored state gets the same number this geometry stored.
    # ``decompose_on_scop_resolved_range`` explains why restricting the matrix
    # in place is not sufficient for that.
    if scop_resolved_range_basis(scop_states, centered.shape[0]) is not None:
        resolved_decomposition = decompose_on_scop_resolved_range(centered, scop_states)
        log_pdet = resolved_decomposition.log_pdet
        hessian_rank = resolved_decomposition.rank

    return SCOPJointGeometry(
        centered_hessian=_readonly(centered),
        hessian_inverse=_readonly(hessian_inverse),
        transformed_intercept_cross=_readonly(transformed_cross),
        sum_w=float(fisher_sum_w),
        log_det_H=float(np.log(fisher_sum_w) + log_pdet),
        hessian_rank=1 + hessian_rank,
        curvature_source=curvature_source,
        transformed_mean_x=_readonly(transformed_mean),
    )


def install_scop_postfit_inference(
    result: PIRLSResult,
    *,
    raw_fisher_gram: NDArray,
    fisher_xtw: NDArray,
    fisher_sum_w: float,
    latent_penalty: NDArray,
    scop_states: Mapping[int, dict],
    groups: Sequence[GroupSlice],
    observed_geometry: SCOPJointGeometry,
    centered_fisher_gram: NDArray | None = None,
    fisher_mean_x: NDArray | None = None,
    dm: DesignMatrix | None = None,
    fisher_weights: FisherWeights | None = None,
) -> SCOPInferenceInfo:
    """Install one terminal SCOP inference state on a mutable solver result."""
    inference = build_scop_postfit_inference(
        raw_fisher_gram=raw_fisher_gram,
        fisher_xtw=fisher_xtw,
        fisher_sum_w=fisher_sum_w,
        latent_penalty=latent_penalty,
        scop_states=scop_states,
        groups=groups,
        observed_geometry=observed_geometry,
        centered_fisher_gram=centered_fisher_gram,
        fisher_mean_x=fisher_mean_x,
        dm=dm,
        fisher_weights=fisher_weights,
    )
    result.scop_geometry = observed_geometry
    result.scop_inference = inference
    result.log_det_H = observed_geometry.log_det_H
    result.reml_hessian_rank = observed_geometry.hessian_rank
    result.effective_df = inference.total_edf
    if result.rank_info is not None:
        result.rank_info = replace(
            result.rank_info,
            intercept_edf=inference.intercept_edf,
            feature_edf=inference.feature_edf,
            group_edf=inference.group_edf,
        )
    return inference


def assemble_observed_scop_hessian(
    *,
    raw_observed_gram: NDArray,
    raw_negative_score: NDArray,
    penalty: NDArray,
    scop_states: dict[int, dict],
    XtW1: NDArray,
    sum_W: float,
) -> tuple[NDArray, NDArray]:
    """Return Pya--Wood's profiled observed Hessian in latent coordinates.

    ``raw_negative_score`` is the slope gradient of the negative
    unit-dispersion log likelihood in mapped coefficient coordinates.  For a
    SCOP block with an elementwise coefficient map, the diagonal block is

    ``J X' W_obs X J + diag(F'' X' g_eta) + S_lambda``.

    Ordinary/SCOP and SCOP/SCOP cross-blocks, followed by the intercept Schur
    complement, are delegated to the shared joint-assembly kernel.
    """
    raw_observed_gram = np.asarray(raw_observed_gram, dtype=np.float64)
    raw_negative_score = np.asarray(raw_negative_score, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    if raw_observed_gram.ndim != 2 or raw_observed_gram.shape[0] != raw_observed_gram.shape[1]:
        raise ValueError("raw_observed_gram must be square")
    width = raw_observed_gram.shape[0]
    if raw_negative_score.shape != (width,):
        raise ValueError("raw_negative_score must match the slope dimension")
    if penalty.shape != (width, width):
        raise ValueError("penalty must match the slope dimension")
    if not (
        np.all(np.isfinite(raw_observed_gram))
        and np.all(np.isfinite(raw_negative_score))
        and np.all(np.isfinite(penalty))
    ):
        raise ValueError("SCOP observed geometry inputs must be finite")

    raw_observed_gram = 0.5 * (raw_observed_gram + raw_observed_gram.T)
    penalty = 0.5 * (penalty + penalty.T)
    joint_states: dict[int, dict] = {}
    transformed_cross = np.asarray(XtW1, dtype=np.float64).copy()
    if transformed_cross.shape != (width,):
        raise ValueError("XtW1 must match the slope dimension")

    for group_index, state in scop_states.items():
        group_slice = state["group_sl"]
        jacobian = _jacobian_diag(state)
        if group_slice.stop - group_slice.start != len(jacobian):
            raise ValueError("SCOP Jacobian width does not match its coefficient slice")
        block = raw_observed_gram[group_slice, group_slice] * jacobian[:, None] * jacobian[None, :]
        block = block.copy()
        second_derivative = _second_derivative_diag(state)
        block[np.diag_indices_from(block)] += second_derivative * raw_negative_score[group_slice]
        block += penalty[group_slice, group_slice]
        joint_state = dict(state)
        joint_state["H_scop_penalized"] = block
        joint_states[group_index] = joint_state
        transformed_cross[group_slice] *= jacobian

    # Local import avoids a module cycle when the SCOP optimizer selects this
    # observed-geometry builder.
    from superglm.reml.scop_efs import assemble_joint_hessian

    centered, _ = assemble_joint_hessian(
        raw_observed_gram + penalty,
        joint_states,
        XtW1=XtW1,
        sum_W=sum_W,
    )
    return centered, transformed_cross


def build_observed_scop_joint_geometry(
    *,
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    result: PIRLSResult,
    penalty: NDArray,
    scop_states: dict[int, dict],
    fisher_XtWX: NDArray | None = None,
    fisher_XtW1: NDArray | None = None,
    fisher_sum_W: float | None = None,
    centered_fisher_gram: NDArray | None = None,
    fisher_mean_x: NDArray | None = None,
    eta_unclipped: NDArray | None = None,
) -> SCOPJointGeometry:
    """Build and decompose the retained mode's exact observed SCOP geometry.

    If the observed latent Hessian is materially indefinite, the optional
    Fisher inputs provide Pya--Wood's positive expected-curvature fallback.
    The returned source and matrix always describe the same decomposition.
    """
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    offset_arr = np.asarray(offset_arr, dtype=np.float64)
    if y.shape != (dm.n,) or sample_weight.shape != y.shape or offset_arr.shape != y.shape:
        raise ValueError("SCOP geometry rows must match the design")
    penalty = np.asarray(penalty, dtype=np.float64)
    if penalty.shape != (dm.p, dm.p):
        raise ValueError("SCOP geometry penalty must match the design")

    if eta_unclipped is None:
        eta_raw = dm.matvec(result.beta) + result.intercept + offset_arr
    else:
        eta_raw = np.asarray(eta_unclipped, dtype=np.float64)
        if eta_raw.shape != y.shape or not np.all(np.isfinite(eta_raw)):
            raise ValueError("retained SCOP eta must be finite and match the design rows")
    eta = stabilize_eta(eta_raw, link)
    mu = clip_mu(link.inverse(eta), distribution)
    observed_weights = compute_scop_observed_information_weights(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
    )
    observed_sum_w = float(np.sum(observed_weights, dtype=np.float64))
    if not np.isfinite(observed_sum_w) or observed_sum_w <= 0.0:
        raise ValueError("SCOP observed intercept curvature must be positive and finite")
    observed_centered = build_centered_system(
        dm=dm,
        W=observed_weights,
        z_off=np.zeros(dm.n, dtype=np.float64),
        penalty=np.zeros_like(penalty),
        tabmat_split=dm.tabmat_centering_split,
    )
    observed_scale = np.sqrt(np.maximum(np.diag(observed_centered.data_gram), 0.0) / observed_sum_w)
    if not _raw_centering_well_scaled(observed_centered.mean_x, observed_scale):
        observed_centered = build_anchor_centered_system(
            dm=dm,
            W=observed_weights,
            z_off=np.zeros(dm.n, dtype=np.float64),
            penalty=np.zeros_like(penalty),
        )

    variance = np.maximum(
        np.asarray(distribution.variance(mu), dtype=np.float64),
        _VARIANCE_FLOOR,
    )
    dmu_deta = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
    fisher_weights = sample_weight * dmu_deta**2 / variance
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        negative_score_eta = sample_weight * (mu - y) * dmu_deta / variance
    if not np.all(np.isfinite(negative_score_eta)):
        raise ValueError("SCOP observed score rows must be finite")
    raw_negative_score = dm.rmatvec(negative_score_eta)
    jacobian = _joint_jacobian_diag(dm.p, scop_states)
    transformed_mean = observed_centered.mean_x * jacobian
    transformed_cross = observed_sum_w * transformed_mean
    centered = observed_centered.data_gram * jacobian[:, None] * jacobian[None, :] + penalty
    for state in scop_states.values():
        group_slice = state["group_sl"]
        second_derivative = _second_derivative_diag(state)
        block = centered[group_slice, group_slice].copy()
        block[np.diag_indices_from(block)] += second_derivative * raw_negative_score[group_slice]
        centered[group_slice, group_slice] = block
    curvature_source: Literal["observed", "fisher"] = "observed"
    sum_w = observed_sum_w
    try:
        decomposition = decompose_gram(centered)
    except ValueError:
        if fisher_XtWX is None or fisher_XtW1 is None or fisher_sum_W is None:
            raise
        fisher_XtWX = np.asarray(fisher_XtWX, dtype=np.float64)
        fisher_XtW1 = np.asarray(fisher_XtW1, dtype=np.float64)
        if (centered_fisher_gram is None) != (fisher_mean_x is None):
            raise ValueError("centered_fisher_gram and fisher_mean_x must be provided together")
        if fisher_mean_x is None:
            fisher_mean = fisher_XtW1 / float(fisher_sum_W)
            fisher_centered = fisher_XtWX - (
                np.outer(fisher_XtW1, fisher_XtW1) / float(fisher_sum_W)
            )
        else:
            fisher_mean = np.asarray(fisher_mean_x, dtype=np.float64)
            fisher_centered = np.asarray(centered_fisher_gram, dtype=np.float64)
            if fisher_mean.shape != (dm.p,) or fisher_centered.shape != (dm.p, dm.p):
                raise ValueError("stable Fisher geometry must match the design")
        centered = fisher_centered * jacobian[:, None] * jacobian[None, :] + penalty
        transformed_mean = fisher_mean * jacobian
        transformed_cross = float(fisher_sum_W) * transformed_mean
        decomposition = _decompose_with_factor_certification(
            centered,
            certifier=_factor_certifier(
                dm=dm,
                fisher_weights=fisher_weights,
                jacobian=jacobian,
                latent_penalty=penalty,
            ),
            center=fisher_mean,
        )
        curvature_source = "fisher"
        sum_w = float(fisher_sum_W)

    return SCOPJointGeometry(
        centered_hessian=_readonly(centered),
        hessian_inverse=_readonly(decomposition.pseudo_inverse()),
        transformed_intercept_cross=_readonly(transformed_cross),
        sum_w=sum_w,
        log_det_H=float(np.log(sum_w) + decomposition.log_pdet),
        hessian_rank=1 + decomposition.rank,
        curvature_source=curvature_source,
        transformed_mean_x=_readonly(transformed_mean),
    )
