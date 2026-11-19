"""Covariance and inference-state computations for fitted models."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from superglm._weight_units import coefficient_covariance, scale_penalties, solver_weights
from superglm.group_matrix import DesignMatrix
from superglm.inference._metrics_design import MappedColumnFactor, factor_from_gram
from superglm.inference.covariance import (
    StructuredCovarianceAccessor,
    StructuredSlopeCovarianceAccessor,
)
from superglm.model.fit_state import fitted_lambda2, fitted_penalty
from superglm.model.retired_state import retained_linear_state
from superglm.solvers._structured.balance_tree import (
    ProfiledSumToZeroTreeFactor,
)
from superglm.solvers._structured.geometry import (
    centered_operator_coefficient_estimable,
)
from superglm.solvers._structured.operators import (
    CenteredBlockOperator,
)
from superglm.solvers._structured.state import (
    StructuredLinearSystemState,
)
from superglm.solvers.identifiability import (
    Identifiability,
    decide_identifiability,
    penalty_null_space,
)
from superglm.solvers.rank import (
    decide_data_rank,
    decompose_factor,
    decompose_gram,
    decompose_gram_if_authoritative,
    diagonal_of_square,
    selected_group_name_set,
)
from superglm.solvers.working_rows import fisher_working_weights
from superglm.types import GroupSlice


def _solver_space_working_weights(model) -> NDArray:
    """Working weights computed against the solver-space fit state."""
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.solvers.mode_score import linear_predictor

    solver = model._solver_pirls_result()
    # the fitted mode's own eta, from its centred state when it carries one
    eta = stabilize_eta(linear_predictor(model._dm, solver, model._fit_offset), model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    return fisher_working_weights(
        distribution=model._distribution,
        link=model._link,
        mu=mu,
        eta=eta,
        sample_weight=solver_weights(model._fit_weights, solver),
    )


def _public_intercept_shift(model, active_groups, p_active: int) -> NDArray:
    """Return the solver-to-public intercept shift in active coordinates."""
    runtime_state = getattr(model, "_runtime_canonical_state", None)
    intercept_shift = np.zeros(p_active, dtype=np.float64)
    if runtime_state is None or p_active <= 0:
        return intercept_shift
    for term_state in runtime_state.get("terms", {}).values():
        if not term_state.get("applied_to_public_model", False):
            continue
        for group_state in term_state.get("groups", []):
            active_group = next(
                (group for group in active_groups if group.name == group_state["group_name"]),
                None,
            )
            if active_group is None:
                continue
            column_means = np.asarray(group_state["column_means"], dtype=np.float64).ravel()
            intercept_shift[active_group.sl] = column_means
    return intercept_shift


def _public_augmented_covariance(model, XtWX_inv_aug: NDArray, active_groups) -> NDArray:
    """Map solver-space augmented covariance into the public runtime state."""
    p_active = XtWX_inv_aug.shape[0] - 1
    intercept_shift = _public_intercept_shift(model, active_groups, p_active)

    if not np.any(intercept_shift):
        return XtWX_inv_aug

    feature_covariance = XtWX_inv_aug[1:, 1:]
    original_cross = XtWX_inv_aug[0, 1:].copy()
    shifted_cross = original_cross + intercept_shift @ feature_covariance
    result = np.array(XtWX_inv_aug, dtype=np.float64, copy=True)
    result[0, 0] = (
        XtWX_inv_aug[0, 0]
        + 2.0 * float(intercept_shift @ original_cross)
        + float(intercept_shift @ feature_covariance @ intercept_shift)
    )
    result[0, 1:] = shifted_cross
    result[1:, 0] = shifted_cross
    return result


def _grouped_active_state(model, selected_names: set[str]):
    """Return a compact grouped design and its original selected columns."""
    active_groups: list[GroupSlice] = []
    active_matrices = []
    rebuilt_columns: list[int] = []
    col = 0
    for gm, group in zip(model._dm.group_matrices, model._groups, strict=True):
        if group.name not in selected_names:
            continue
        active_matrices.append(gm)
        rebuilt_columns.extend(range(group.start, group.end))
        active_groups.append(replace(group, start=col, end=col + group.size))
        col += group.size
    return (
        DesignMatrix(active_matrices, n=model._dm.n, p=col),
        active_groups,
        np.asarray(rebuilt_columns, dtype=np.intp),
    )


def _rank_active_state(model, rank_info):
    """Return the explicitly selected grouped design in rank-info order."""
    design, active_groups, rebuilt_columns = _grouped_active_state(
        model,
        set(rank_info.selected_group_names),
    )
    if not np.array_equal(
        rebuilt_columns,
        np.asarray(rank_info.selected_columns, dtype=np.intp),
    ):
        raise ValueError("rank metadata selected columns do not match active groups")
    return design, active_groups


def _legacy_rank_state(model, solver, W: NDArray):
    """The selected grouped design, its centred system at ``W`` and their rank decisions.

    Returns ``design, active_groups, centered, coefficient_rank,
    profile_rank, data_rank``: the raw and centred Hessians' decompositions,
    decided as the fit decides them, and the data Gram's where it is the
    centred Hessian (unpenalized; ``None`` otherwise, computed on demand by
    ``fit_data_rank``), and ``None`` for the last four when no column is
    selected.
    """
    from superglm.inference.covariance import _active_penalty_matrix
    from superglm.solvers.centered_system import (
        build_centered_system,
        grouped_augmented_factor,
        grouped_weighted_factor,
    )

    selected_names = selected_group_name_set(
        solver,
        model._groups,
        penalty=fitted_penalty(model),
    )
    design, active_groups, _ = _grouped_active_state(model, selected_names)
    lam2 = scale_penalties(fitted_lambda2(model), getattr(solver, "weight_exponent", 0))
    curvature = np.array(
        _active_penalty_matrix(
            model._dm.group_matrices,
            model._groups,
            active_groups,
            lam2,
            reml_penalties=getattr(model, "_reml_penalties", None),
        ),
        dtype=np.float64,
        copy=True,
    )
    from superglm.solvers.pirls import _add_selection_local_curvature

    original_by_name = {group.name: group for group in model._groups}
    original_active_groups = [original_by_name[group.name] for group in active_groups]
    _add_selection_local_curvature(
        curvature=curvature,
        penalty=fitted_penalty(model),
        beta=np.asarray(solver.beta, dtype=np.float64),
        original_groups=original_active_groups,
        active_groups=active_groups,
    )
    if design.p == 0:
        return design, active_groups, None, None, None, None

    centered = build_centered_system(
        dm=design,
        W=W,
        z_off=np.zeros(design.n),
        penalty=curvature,
    )
    raw_gram, _, _, _ = centered.raw_weighted_moments()
    coefficient_rank = decompose_gram_if_authoritative(raw_gram + curvature)
    # rows centred about the system's exact centre pair
    centre, centre_lo = centered.centre_pair()
    if coefficient_rank is None:
        certified = decompose_factor(grouped_augmented_factor(design, W, curvature))
        coefficient_rank = certified
    data_rank = None
    if not np.any(curvature):
        # The fit's route (``rank.decide_data_rank``): the data rank is the
        # profile rank, so the full factor decides it, as in the fit.
        data_rank = decide_data_rank(
            centered.data_gram,
            lambda: decompose_factor(
                grouped_weighted_factor(design, W, center=centre, center_lo=centre_lo)
            ),
            rows=design.n,
            certify=True,
        )
        profile_rank = data_rank
    else:
        profile_rank = decompose_gram_if_authoritative(centered.data_gram + curvature)
        if profile_rank is None:
            certified = decompose_factor(
                grouped_augmented_factor(
                    design,
                    W,
                    curvature,
                    center=centre,
                    center_lo=centre_lo,
                )
            )
            profile_rank = certified
    return design, active_groups, centered, coefficient_rank, profile_rank, data_rank


def _legacy_active_state(model, solver, W: NDArray):
    """Rebuild old fit inference with grouped, centered coefficient algebra.

    The last value is the decomposition of the centred Hessian whose inverse
    the covariance is: which coordinates it fixes at zero is what the fit
    leaves unidentified (``solvers.identifiability``).
    """
    design, active_groups, centered, coefficient_rank, profile_rank, _ = _legacy_rank_state(
        model, solver, W
    )
    if centered is None:
        sum_w = float(np.sum(W))
        augmented = np.array([[1.0 / sum_w]]) if sum_w > 0.0 else np.array([[0.0]])
        return (
            design,
            active_groups,
            np.empty((0, 0)),
            _public_augmented_covariance(model, augmented, active_groups),
            np.empty((0, 0)),
            np.empty((0, 0)),
            decompose_gram(np.empty((0, 0))),
        )
    _, xtw1, _, _ = centered.raw_weighted_moments()
    coefficient_inverse = coefficient_rank.pseudo_inverse()
    profile_inverse = profile_rank.pseudo_inverse()
    mean_x = xtw1 / centered.sum_w
    intercept_cross = -(profile_inverse @ mean_x)
    augmented = np.empty((design.p + 1, design.p + 1), dtype=np.float64)
    augmented[0, 0] = 1.0 / centered.sum_w + float(mean_x @ profile_inverse @ mean_x)
    augmented[0, 1:] = intercept_cross
    augmented[1:, 0] = intercept_cross
    augmented[1:, 1:] = profile_inverse
    augmented = _public_augmented_covariance(model, augmented, active_groups)
    return (
        design,
        active_groups,
        coefficient_inverse,
        augmented,
        centered.data_gram,
        profile_inverse,
        profile_rank,
    )


def face_record_at(model, W: NDArray):
    """The fit's rank record formed again at the weights ``W``, on the face it recorded.

    The face and its rows are the fit's: a revision that leaves every
    constrained term's coefficients where its QP put them leaves the rows the
    QP held binding. Each decomposition, on the face and off it, the EDF and
    the centre are formed at ``W`` as the fit forms them; the data Gram is
    left for the first inference at ``W`` to take.
    """
    from superglm.solvers.centered_system import grouped_augmented_factor, grouped_projected_factor
    from superglm.solvers.irls_direct import _face_decomposition, _feature_edf
    from superglm.solvers.rank import weights_fingerprint

    solver = model._solver_pirls_result()
    record = solver.rank_info
    design, active_groups, centered, coefficient, augmented, data = _legacy_rank_state(
        model, solver, W
    )
    curvature, face = centered.penalty, record.face
    centre, centre_lo = centered.centre_pair()

    def factor(exponent: int = 0) -> NDArray:
        return grouped_augmented_factor(
            design,
            np.ldexp(W, -exponent),
            np.ldexp(curvature, -exponent),
            center=centre,
            center_lo=centre_lo,
        )

    def projected_rows(exponent: int = 0) -> NDArray:
        return grouped_projected_factor(
            design, np.ldexp(W, -exponent), face, center=centre, center_lo=centre_lo
        )[0]

    face_augmented = face_coefficient = None
    if face is not None:
        face_augmented = _face_decomposition(centered.hessian, face, factor, design.n)
        raw = centered.hessian + centered.sum_w * np.outer(centered.mean_x, centered.mean_x)
        face_coefficient = _face_decomposition(
            raw, face, lambda: grouped_augmented_factor(design, W, curvature), design.n
        )
    feature_edf, feature_edf1 = _feature_edf(
        centered,
        augmented if face is None else face_augmented,
        factor if face is None else projected_rows,
        face,
    )
    feature_edf[np.abs(feature_edf) < 100.0 * np.finfo(float).eps] = 0.0
    feature_edf.setflags(write=False)
    if feature_edf1 is not None:
        feature_edf1.setflags(write=False)
    return replace(
        record,
        sum_w=centered.sum_w,
        mean_x=centered.mean_x,
        data=data,
        identifiability=None,
        augmented=augmented,
        coefficient=coefficient,
        feature_edf=feature_edf,
        group_edf={group.name: float(np.sum(feature_edf[group.sl])) for group in active_groups},
        data_gram=centered.data_gram,
        data_gram_weights=weights_fingerprint(W),
        face_augmented=face_augmented,
        face_coefficient=face_coefficient,
        feature_edf1=feature_edf1,
    )


def _rank_centered_data_gram(X_active: DesignMatrix, W: NDArray) -> NDArray:
    """Build the selected centered Gram without a full observation-row matrix."""
    from superglm.solvers.centered_system import build_centered_system

    p_active = X_active.p
    if p_active == 0:
        return np.empty((0, 0))
    return build_centered_system(
        dm=X_active,
        W=W,
        z_off=np.zeros(len(W)),
        penalty=np.zeros((p_active, p_active)),
    ).data_gram


def _rank_edf1(rank_info, data_gram: NDArray, edf: NDArray) -> NDArray:
    """Alternative EDF in the decomposition's retained coefficient space.

    On a face the factor decided, the record holds it, read off the face's
    projected rows with the EDF; the formed Gram is not read there.
    """
    import scipy.linalg

    if rank_info.face is not None and rank_info.feature_edf1 is not None:
        return rank_info.feature_edf1[rank_info.selected_columns].copy()
    decomposition = rank_info.augmented
    if decomposition.rank == 0:
        return np.zeros_like(edf)

    if decomposition.cholesky_factor is not None and rank_info.face is None:
        active = decomposition.active_columns
        scale = decomposition.column_scale[active]
        retained_gram = data_gram[np.ix_(active, active)] / np.outer(scale, scale)
        retained_inverse = scipy.linalg.cho_solve(
            (decomposition.cholesky_factor, True),
            np.eye(decomposition.rank),
            check_finite=False,
        )
        retained_F = retained_inverse @ retained_gram
        edf1 = 2.0 * edf
        edf1[active] -= diagonal_of_square(retained_F)
        return edf1

    F = rank_info.augmented_inverse() @ data_gram
    return 2.0 * edf - diagonal_of_square(F)


def _rank_augmented_covariance(model, rank_info, active_groups):
    """Transform centered retained covariance to solver and public intercepts."""
    p_active = len(rank_info.selected_columns)
    feature_covariance = rank_info.covariance_inverse()
    mean_x = rank_info.mean_x[rank_info.selected_columns]
    intercept_cross = -(feature_covariance @ mean_x)
    solver_covariance = np.empty((p_active + 1, p_active + 1), dtype=np.float64)
    solver_covariance[0, 0] = (1.0 / rank_info.sum_w if rank_info.sum_w > 0.0 else 0.0) + float(
        mean_x @ feature_covariance @ mean_x
    )
    solver_covariance[0, 1:] = intercept_cross
    solver_covariance[1:, 0] = intercept_cross
    solver_covariance[1:, 1:] = feature_covariance
    return _public_augmented_covariance(model, solver_covariance, active_groups)


def _penalty_null_spaces(model, active_groups) -> list[NDArray]:
    """Each selected group's penalty null space ``N_S,g``, without a smoothing parameter in it.

    A group with REML components is balanced over the components whose fitted lambda is
    positive (``identifiability.penalty_null_space``); any other group's penalty is the fitted
    curvature's block (one lambda each, and the selection penalty's local curvature), as the
    fit's Hessian holds it.
    """
    from superglm.inference.covariance import _active_penalty_matrix
    from superglm.reml.penalty_algebra import (
        _penalty_component_omega_ssp,
        penalty_component_dense_matrix,
    )
    from superglm.solvers.pirls import _add_selection_local_curvature

    solver = model._solver_pirls_result()
    lam2 = scale_penalties(fitted_lambda2(model), getattr(solver, "weight_exponent", 0))
    components: dict[str, list] = {}
    for component in getattr(model, "_reml_penalties", None) or ():
        lam = lam2.get(component.name, 0.0) if isinstance(lam2, dict) else lam2
        if lam > 0.0:
            components.setdefault(component.group_name, []).append(component)
    matrices = {g.name: gm for g, gm in zip(model._groups, model._dm.group_matrices, strict=True)}
    # the groups without REML components, in their own compact coordinates
    missing, column = [], 0
    for group in active_groups:
        if group.name not in components:
            missing.append(dataclasses.replace(group, start=column, end=column + group.size))
            column += group.size
    fitted: dict[str, NDArray] = {}
    if missing:
        curvature = np.array(
            _active_penalty_matrix(
                model._dm.group_matrices,
                model._groups,
                missing,
                lam2,
                reml_penalties=getattr(model, "_reml_penalties", None),
            ),
            dtype=np.float64,
            copy=True,
        )
        original = {group.name: group for group in model._groups}
        _add_selection_local_curvature(
            curvature=curvature,
            penalty=fitted_penalty(model),
            beta=np.asarray(solver.beta, dtype=np.float64),
            original_groups=[original[group.name] for group in missing],
            active_groups=missing,
        )
        fitted = {group.name: curvature[group.sl, group.sl] for group in missing}
    spaces = []
    for group in active_groups:
        own = components.get(group.name)
        if own is None:
            spaces.append(penalty_null_space([fitted[group.name]], group.size))
            continue
        kinds = {component.penalty_kind for component in own}
        widths = {component.block_width for component in own}
        if "identity" in kinds:
            # a ridge on every coordinate
            spaces.append(np.zeros((group.size, 0)))
        elif kinds <= {"repeated", "sum_to_zero"} and len(widths) == 1:
            # ``I kron Omega`` or ``C'C kron Omega``, ``C'C`` positive definite: the null space
            # is ``I kron null(Omega)``, found without expanding the term's penalty
            width = int(next(iter(widths)))
            omegas = [_penalty_component_omega_ssp(c, matrices[group.name]) for c in own]
            local = penalty_null_space([omega for omega in omegas if omega is not None], width)
            spaces.append(np.kron(np.eye(group.size // width), local))
        else:
            blocks = [penalty_component_dense_matrix(c, matrices[group.name]) for c in own]
            spaces.append(penalty_null_space(blocks, group.size))
    return spaces


def _decided_identifiability(
    model, decomposition, active_groups, design, W, data_gram, *, penalized=True
) -> Identifiability:
    """The identifiability of a fit whose centred Hessian is ``decomposition``, in fitted coordinates.

    ``decomposition`` is in the coordinates of ``active_groups`` (the selected groups), ``design``
    their rows; ``solvers.identifiability`` decides and explains the record.
    """
    original = {group.name: group for group in model._groups}
    columns = np.zeros(0, dtype=np.intp)
    if active_groups:
        columns = np.concatenate(
            [np.arange(original[g.name].start, original[g.name].end) for g in active_groups]
        )
    return decide_identifiability(
        decomposition,
        columns=columns,
        width=len(model._solver_pirls_result().beta),
        groups=active_groups,
        null_spaces=(
            _penalty_null_spaces(model, active_groups)
            if penalized
            else [np.eye(group.size) for group in active_groups]
        ),
        data_gram=data_gram,
        rows=design.n,
        design=design,
        specs={**getattr(model, "_specs", {}), **getattr(model, "_interaction_specs", {})},
        W=W,
    )


def _record_identifiability(
    model, rank_info, design, active_groups, W, data_gram
) -> Identifiability | None:
    """The record's identifiability, decided on first inference and kept on the record.

    ``None`` for a record saved before fits recorded it: its standard errors keep their flags
    (``RankInfo.coefficient_identified``).
    """
    if rank_info.identifiability_policy != "penalised":
        return None
    if rank_info.identifiability is None:
        decided = _decided_identifiability(
            model,
            rank_info.augmented,
            active_groups,
            design,
            W,
            data_gram,
            # an unpenalized fit's data rank is its Hessian's (``RankInfo.data``)
            penalized=rank_info.data is not rank_info.augmented,
        )
        object.__setattr__(rank_info, "identifiability", decided)
    return rank_info.identifiability


def fit_data_rank(model):
    """The fit's rank record with its data-only rank, computed on first request and kept on it.

    The data rank answers whether the data alone, ignoring the penalty, determine a direction:
    a diagnostic (the editor's free levels, ``diagnose``) that no standard error reads, so a fit
    no longer computes it. It is decided by the fit's route (``rank.decide_data_rank``) at the
    fit's working weights, and the REML profile records which route decided it
    (``data_rank_route``). ``None`` without a rank record; a record whose design was released
    keeps ``data`` unset.
    """
    from superglm.solvers.centered_system import (
        build_centered_system,
        grouped_projected_factor,
        grouped_structural_nulls,
        grouped_weighted_factor,
    )

    solver = model._solver_pirls_result()
    record = getattr(solver, "rank_info", None)
    if record is None or record.data is not None or model._dm is None:
        return record
    design, _ = _rank_active_state(model, record)
    W = _solver_space_working_weights(model)
    centered = build_centered_system(
        dm=design, W=W, z_off=np.zeros(design.n), penalty=np.zeros((design.p, design.p))
    )
    centre, centre_lo = centered.centre_pair()
    data = decide_data_rank(
        centered.data_gram,
        lambda: decompose_factor(
            grouped_weighted_factor(design, W, center=centre, center_lo=centre_lo)
        ),
        project=lambda basis: grouped_projected_factor(
            design, W, basis, center=centre, center_lo=centre_lo
        ),
        structural=lambda: grouped_structural_nulls(design, W, center=centre, center_lo=centre_lo),
        rows=design.n,
        # Fisher weights are nonnegative, so the data Gram is PSD by construction
        psd_by_construction=True,
        certify=False,
    )
    object.__setattr__(record, "data", data)
    profile = getattr(model, "_reml_profile", None)
    if isinstance(profile, dict):
        # ``"gram"`` (the Gram certified itself), ``"grey_subspace"`` (the Gram plus a row
        # factor of its ``k`` grey directions, ``data_rank_grey_k``) or ``"row_factor"``.
        if data.grey_subspace is not None:
            profile["data_rank_route"] = "grey_subspace"
            profile["data_rank_grey_k"] = int(data.grey_subspace["k"])
        else:
            profile["data_rank_route"] = "row_factor" if data.method == "qr_svd" else "gram"
    return record


def _structured_covariance_state(
    model,
    state: StructuredLinearSystemState,
) -> tuple[
    StructuredSlopeCovarianceAccessor,
    StructuredCovarianceAccessor,
    list[GroupSlice],
]:
    """Build lazy covariance views over one retained structured fit."""
    active_groups = list(model._groups)
    shift = _public_intercept_shift(
        model,
        active_groups,
        state.profiled_factor.shape[0],
    )
    augmented = StructuredCovarianceAccessor(
        state.profiled_factor,
        intercept_shift=shift,
    )
    # The slope covariance M_ss, the slope block of the augmented inverse: no
    # raw-coordinate coefficient factor exists.
    return augmented.slopes, augmented, active_groups


def coef_covariance(model):
    """Phi-scaled Bayesian covariance for active coefficients."""
    solver = model._solver_pirls_result()
    scop_inference = getattr(solver, "scop_inference", None)
    if scop_inference is not None:
        if solver.rank_info is not None:
            _, active_groups = _rank_active_state(model, solver.rank_info)
            selected = np.asarray(solver.rank_info.selected_columns, dtype=np.intp)
        else:
            _, active_groups, selected = _grouped_active_state(
                model,
                {group.name for group in model._groups},
            )
        mapped_covariance = np.asarray(scop_inference.augmented_inverse)[1:, 1:]
        covariance = coefficient_covariance(mapped_covariance[np.ix_(selected, selected)], solver)
        return covariance, active_groups
    linear_state = retained_linear_state(model)
    if isinstance(linear_state, StructuredLinearSystemState):
        _, augmented, active_groups = _structured_covariance_state(model, linear_state)
        return coefficient_covariance(augmented.slopes, solver), active_groups
    if solver.rank_info is not None:
        _, active_groups = _rank_active_state(model, solver.rank_info)
        covariance = coefficient_covariance(solver.rank_info.covariance_inverse(), solver)
        return covariance, active_groups
    W = _solver_space_working_weights(model)
    _, active_groups, _, _, _, profile_inverse, _ = _legacy_active_state(model, solver, W)
    return coefficient_covariance(profile_inverse, solver), active_groups


def fit_active_info(model):
    """Grouped active design, weights, and (X'WX+S)^{-1} from fit state."""
    solver = model._solver_pirls_result()
    # first: a retired state without its design raises the refit message here
    linear_state = retained_linear_state(model)
    W = _solver_space_working_weights(model)
    scop_inference = getattr(solver, "scop_inference", None)
    if scop_inference is not None:
        if solver.rank_info is not None:
            X_active, active_groups = _rank_active_state(model, solver.rank_info)
            selected = np.asarray(solver.rank_info.selected_columns, dtype=np.intp)
        else:
            X_active, active_groups, selected = _grouped_active_state(
                model,
                {group.name for group in model._groups},
            )
        inverse = np.asarray(scop_inference.coefficient_inverse)[np.ix_(selected, selected)]
        augmented_indices = np.concatenate(([0], selected + 1))
        augmented = np.asarray(scop_inference.augmented_inverse)[
            np.ix_(augmented_indices, augmented_indices)
        ]
        augmented = _public_augmented_covariance(model, augmented, active_groups)
        return X_active, W, inverse, augmented, active_groups
    if isinstance(linear_state, StructuredLinearSystemState):
        inverse, augmented, active_groups = _structured_covariance_state(
            model,
            linear_state,
        )
        return model._dm, W, inverse, augmented, active_groups
    if solver.rank_info is not None:
        X_active, active_groups = _rank_active_state(model, solver.rank_info)
        inverse = solver.rank_info.coefficient_inverse()
        augmented = _rank_augmented_covariance(model, solver.rank_info, active_groups)
        return X_active, W, inverse, augmented, active_groups

    X_active, active_groups, inverse, augmented, _, _, _ = _legacy_active_state(
        model,
        solver,
        W,
    )
    return X_active, W, inverse, augmented, active_groups


def _structured_small_data_factor(operator: CenteredBlockOperator) -> MappedColumnFactor:
    """Store the exact centered dense-small Gram factor on mapped columns."""
    raw = operator.raw
    small_indices = np.asarray(raw.small_indices, dtype=np.intp)
    if not len(small_indices):
        return MappedColumnFactor(np.empty((0, 0)), small_indices, operator.shape[0])
    cross = operator.cross[small_indices]
    center = operator.center[small_indices]
    data_gram = (
        raw.A
        - np.outer(cross, center)
        - np.outer(center, cross)
        + operator.total * np.outer(center, center)
    )
    return MappedColumnFactor(factor_from_gram(data_gram), small_indices, operator.shape[0])


def inference_identified(inference: dict) -> NDArray | None:
    """Which coefficients an inference record identifies.

    A record saved before the penalised rule names its data-only flags
    ``coefficient_estimable``; such a model keeps them (``RankInfo.coefficient_identified``).
    """
    return inference.get("coefficient_identified", inference.get("coefficient_estimable"))


def fit_inference_info(model):
    """All coefficient-space inference quantities for model.summary().

    Forms the working weights W and per-group Gram blocks rather than the
    n×p active design, so model.summary() is O(n + p³).

    Returns a dict with:
        W : (n,) working weights
        XtWX_inv : (p_active, p_active) = (X'WX + S)^{-1}
        XtWX_inv_aug : (p_active+1, p_active+1) augmented inverse incl. intercept
        active_groups : list of GroupSlice re-indexed to active columns
        R_a : rectangular factor whose Gram is the required centered X'WX block
        edf : per-coefficient EDF vector
        edf1 : Wood's alternative EDF vector
        group_edf_map : per-group summed EDF dict
    """
    solver = model._solver_pirls_result()
    # first: a retired state without its design raises the refit message here
    linear_state = retained_linear_state(model)
    W = _solver_space_working_weights(model)
    scop_inference = getattr(solver, "scop_inference", None)
    if scop_inference is not None:
        rank_info = solver.rank_info
        if rank_info is not None:
            X_active, active_groups = _rank_active_state(model, rank_info)
            selected = np.asarray(rank_info.selected_columns, dtype=np.intp)
            coefficient_identified = rank_info.coefficient_identified()
        else:
            X_active, active_groups, selected = _grouped_active_state(
                model,
                {group.name for group in model._groups},
            )
            coefficient_identified = np.ones(len(solver.beta), dtype=bool)
        inverse = np.asarray(scop_inference.coefficient_inverse)[np.ix_(selected, selected)]
        augmented_indices = np.concatenate(([0], selected + 1))
        augmented = np.asarray(scop_inference.augmented_inverse)[
            np.ix_(augmented_indices, augmented_indices)
        ]
        augmented = _public_augmented_covariance(model, augmented, active_groups)
        edf = np.asarray(scop_inference.feature_edf)[selected].copy()
        edf1 = np.asarray(scop_inference.feature_edf1)[selected].copy()
        # The fit's Gram, when the SCOP solve left one at these weights; either
        # way the record holds none afterwards.
        data_gram = None if rank_info is None else rank_info.take_data_gram(W)
        if X_active.p == 0:
            R_a = np.empty((0, 0))
        else:
            if data_gram is None:
                data_gram = _rank_centered_data_gram(X_active, W)
            R_a = factor_from_gram(data_gram)
        identifiability = (
            None
            if rank_info is None or X_active.p == 0
            else _record_identifiability(model, rank_info, X_active, active_groups, W, data_gram)
        )
        return {
            "W": W,
            "XtWX_inv": inverse,
            "XtWX_inv_aug": augmented,
            "active_groups": active_groups,
            "R_a": R_a,
            "edf": edf,
            "edf1": edf1,
            "group_edf_map": dict(scop_inference.group_edf),
            "identifiability": identifiability,
            "coefficient_identified": (
                coefficient_identified
                if identifiability is None
                else identifiability.coefficient_identified()
            ),
        }
    if isinstance(linear_state, StructuredLinearSystemState):
        inverse, augmented, active_groups = _structured_covariance_state(
            model,
            linear_state,
        )
        edf = linear_state.profiled_factor.inverse_operator_diagonal(
            linear_state.centered_data_operator,
        )
        edf[np.abs(edf) < 100.0 * np.finfo(float).eps] = 0.0
        edf1 = 2.0 * edf - linear_state.profiled_factor.inverse_operator_square_diagonal(
            linear_state.centered_data_operator,
        )
        group_edf_map = {group.name: float(np.sum(edf[group.sl])) for group in active_groups}
        return {
            "W": W,
            "XtWX_inv": inverse,
            "XtWX_inv_aug": augmented,
            "active_groups": active_groups,
            "R_a": _structured_small_data_factor(linear_state.centered_data_operator),
            "edf": edf,
            "edf1": edf1,
            "group_edf_map": group_edf_map,
            # The structured solvers still flag the data's null space (an sz
            # factor's on the balance tree, SumToZeroTreeFactor.coefficient_estimable);
            # their fits move to the penalised rule with their own solvers.
            "coefficient_identified": (
                linear_state.profiled_factor.coefficient_estimable()
                if isinstance(linear_state.profiled_factor, ProfiledSumToZeroTreeFactor)
                else centered_operator_coefficient_estimable(linear_state.centered_data_operator)
            ),
            "structured_covariance": True,
        }
    if solver.rank_info is not None:
        rank_info = solver.rank_info
        X_active, active_groups = _rank_active_state(model, rank_info)
        inverse = rank_info.coefficient_inverse()
        augmented = _rank_augmented_covariance(model, rank_info, active_groups)
        # The fit formed this Gram to decide the data rank, at the weights it
        # ended on; only a record without it forms the Gram again.
        data_gram = rank_info.take_data_gram(W)
        if X_active.shape[1] == 0:
            return {
                "W": W,
                "XtWX_inv": inverse,
                "XtWX_inv_aug": augmented,
                "active_groups": active_groups,
                "R_a": np.empty((0, 0)),
                "edf": np.array([]),
                "edf1": np.array([]),
                "group_edf_map": dict(rank_info.group_edf),
                "coefficient_identified": rank_info.coefficient_identified(),
                "identifiability": None,
            }
        if data_gram is None:
            data_gram = _rank_centered_data_gram(X_active, W)
        edf = rank_info.feature_edf[rank_info.selected_columns].copy()
        edf1 = _rank_edf1(rank_info, data_gram, edf)
        R_a = factor_from_gram(data_gram)
        return {
            "W": W,
            "XtWX_inv": inverse,
            "XtWX_inv_aug": augmented,
            "active_groups": active_groups,
            "R_a": R_a,
            "edf": edf,
            "edf1": edf1,
            "group_edf_map": dict(rank_info.group_edf),
            "identifiability": _record_identifiability(
                model, rank_info, X_active, active_groups, W, data_gram
            ),
            # after the record's identifiability, which it reads
            "coefficient_identified": rank_info.coefficient_identified(),
        }

    X_active, active_groups, inverse, augmented, data_gram, profile_inverse, profile_rank = (
        _legacy_active_state(model, solver, W)
    )
    if X_active.p == 0:
        return {
            "W": W,
            "XtWX_inv": inverse,
            "XtWX_inv_aug": augmented,
            "active_groups": active_groups,
            "R_a": np.empty((0, 0)),
            "edf": np.array([]),
            "edf1": np.array([]),
            "group_edf_map": {},
            "coefficient_identified": np.zeros(len(solver.beta), dtype=bool),
            "identifiability": None,
        }

    F = profile_inverse @ data_gram
    edf = np.diag(F)
    edf1 = 2.0 * edf - diagonal_of_square(F)

    R_a = factor_from_gram(data_gram)

    group_edf_map: dict[str, float] = {}
    for ag in active_groups:
        group_edf_map[ag.name] = float(np.sum(edf[ag.sl]))
    identifiability = _decided_identifiability(
        model, profile_rank, active_groups, X_active, W, data_gram
    )

    return {
        "W": W,
        "XtWX_inv": inverse,
        "XtWX_inv_aug": augmented,
        "active_groups": active_groups,
        "R_a": R_a,
        "edf": edf,
        "edf1": edf1,
        "group_edf_map": group_edf_map,
        "coefficient_identified": identifiability.coefficient_identified(),
        "identifiability": identifiability,
    }


def group_edf(model) -> dict[str, float] | None:
    """Per-group effective degrees of freedom via F = (X'WX+S)^{-1} X'WX."""
    if model._dm is None or model._result is None:
        return None
    return cast(dict[str, float] | None, model._fit_inference_info["group_edf_map"])
