"""Covariance and inference-state computations for fitted models."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import SQRT_EPS, gamma_up
from superglm._weight_units import coefficient_covariance, scale_penalties, solver_weights
from superglm.group_matrix import DesignMatrix
from superglm.inference._metrics_design import MappedColumnFactor, factor_from_gram
from superglm.inference.covariance import (
    StructuredCovarianceAccessor,
    StructuredSlopeCovarianceAccessor,
)
from superglm.model.fit_state import fitted_lambda2, fitted_penalty
from superglm.model.retired_state import retained_linear_state
from superglm.solvers._structured.balance_tree import SumToZeroPenalizedOperator
from superglm.solvers._structured.nested import NestedDataOperator, NestedPenalizedOperator
from superglm.solvers._structured.operators import (
    CenteredBlockOperator,
    compact_operator_diagonal,
)
from superglm.solvers._structured.state import (
    StructuredLinearSystemState,
)
from superglm.solvers.identifiability import (
    ConstructionColumns,
    Identifiability,
    TruncatedDecomposition,
    certified_identifiability,
    decide_identifiability,
    penalty_null_space,
    published_edf,
)
from superglm.solvers.rank import (
    RankDecomposition,
    RankInfo,
    decide_data_rank,
    decompose_factor,
    decompose_gram,
    decompose_gram_if_authoritative,
    diagonal_of_square,
    selected_group_name_set,
    weights_fingerprint,
)
from superglm.solvers.working_rows import fisher_working_weights
from superglm.types import GroupSlice

if TYPE_CHECKING:
    from superglm.model.separation_ops import WithheldFit


def _solver_space_working_weights(model) -> NDArray:
    """Working weights computed against the solver-space fit state.

    Rows a certified separation withholds carry no prior weight (``separation_ops``).
    """
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.separation_ops import withheld_weights
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
        sample_weight=solver_weights(withheld_weights(model), solver),
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


def _rank_curvature(model, active_groups, solver=None) -> NDArray:
    """The fitted penalty curvature on ``active_groups``, as the fit's centred Hessian holds it.

    The penalty at the fitted lambdas in the solver's weight unit, and the selection penalty's
    local curvature at the fitted coefficients (``solver``'s, the fit's by default).
    """
    from superglm.inference.covariance import _active_penalty_matrix
    from superglm.solvers.pirls import _add_selection_local_curvature

    solver = model._solver_pirls_result() if solver is None else solver
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
    original_by_name = {group.name: group for group in model._groups}
    _add_selection_local_curvature(
        curvature=curvature,
        penalty=fitted_penalty(model),
        beta=np.asarray(solver.beta, dtype=np.float64),
        original_groups=[original_by_name[group.name] for group in active_groups],
        active_groups=active_groups,
    )
    return curvature


def _legacy_rank_state(model, solver, W: NDArray):
    """The selected grouped design, its centred system at ``W`` and their rank decisions.

    Returns ``design, active_groups, centered, coefficient_rank,
    profile_rank, data_rank``: the raw and centred Hessians' decompositions,
    decided as the fit decides them, and the data Gram's where it is the
    centred Hessian (unpenalized; ``None`` otherwise, computed on demand by
    ``fit_data_rank``), and ``None`` for the last four when no column is
    selected.
    """
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
    curvature = _rank_curvature(model, active_groups, solver)
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
    projected rows with the EDF; the formed Gram is not read there.  So does a
    fit published with its identifiability record (``publish_identified_fit``).
    """
    import scipy.linalg

    record = getattr(rank_info, "identifiability", None)
    if rank_info.face is not None and rank_info.feature_edf1 is not None:
        return rank_info.feature_edf1[rank_info.selected_columns].copy()
    if record is not None and record.edf1 is not None:
        return record.edf1[rank_info.selected_columns].copy()
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
    """Transform centered retained covariance to solver and public intercepts.

    A fit published with weak directions adds their term in low-rank form
    (``Identifiability.weak_root``): the intercept's variance ``c'Vc`` read through the dense
    ``1 / C`` entries would carry their rounding.
    """
    p_active = len(rank_info.selected_columns)
    record = getattr(rank_info, "identifiability", None)
    root = None if record is None else record.weak_root(rank_info.selected_columns)
    feature_covariance = (
        rank_info.covariance_inverse() if root is None else rank_info.covariance_inverse(weak=False)
    )
    mean_x = rank_info.mean_x[rank_info.selected_columns]
    intercept_cross = -(feature_covariance @ mean_x)
    solver_covariance = np.empty((p_active + 1, p_active + 1), dtype=np.float64)
    solver_covariance[0, 0] = (1.0 / rank_info.sum_w if rank_info.sum_w > 0.0 else 0.0) + float(
        mean_x @ feature_covariance @ mean_x
    )
    solver_covariance[0, 1:] = intercept_cross
    solver_covariance[1:, 0] = intercept_cross
    solver_covariance[1:, 1:] = feature_covariance
    if root is not None:
        solver_covariance += root @ root.T
    return _public_augmented_covariance(model, solver_covariance, active_groups)


def _construction_columns(model, active_groups) -> Callable[[int], ConstructionColumns | None]:
    """Each selected group's construction columns of ``X_F``, formed when the gauge reads them.

    A group's penalty null space ``N_S,g`` holds them, with no smoothing parameter in it: a group
    with REML components is balanced over the components whose fitted lambda is positive
    (``identifiability.penalty_null_space``), any other group's penalty is the fitted curvature's
    block (one lambda each, and the selection penalty's local curvature).  A ridge leaves none;
    no penalty leaves the group's coordinates, in order.  A factor smooth's ``I kron Omega`` or
    ``C'C kron Omega`` penalty repeats one level's null space per level (``C'C`` positive
    definite), found without expanding the term.  A smooth's columns are ordered and projected in
    its raw basis (``_raw_construction``).
    """
    from superglm.reml.penalty_algebra import (
        _penalty_component_omega_ssp,
        penalty_component_dense_matrix,
    )

    solver = model._solver_pirls_result()
    lam2 = scale_penalties(fitted_lambda2(model), getattr(solver, "weight_exponent", 0))
    components: dict[str, list] = {}
    for component in getattr(model, "_reml_penalties", None) or ():
        lam = lam2.get(component.name, 0.0) if isinstance(lam2, dict) else lam2
        if lam > 0.0:
            components.setdefault(component.group_name, []).append(component)
    matrices = {g.name: gm for g, gm in zip(model._groups, model._dm.group_matrices, strict=True)}
    built: dict[int, ConstructionColumns | None] = {}

    def columns(index: int) -> ConstructionColumns | None:
        if index in built:
            return built[index]
        group = active_groups[index]
        matrix = matrices[group.name]
        own = components.get(group.name)
        blocks = 1
        if own is None:
            local = dataclasses.replace(group, start=0, end=group.size)
            block = _rank_curvature(model, [local])
            null = penalty_null_space([block], group.size)
        elif "identity" in {component.penalty_kind for component in own}:
            # a ridge on every coordinate
            null = np.zeros((group.size, 0))
        elif {c.penalty_kind for c in own} <= {"repeated", "sum_to_zero"} and len(
            {c.block_width for c in own}
        ) == 1:
            width = int(own[0].block_width)
            omegas = [_penalty_component_omega_ssp(c, matrix) for c in own]
            null = penalty_null_space([omega for omega in omegas if omega is not None], width)
            blocks = group.size // width
        else:
            null = penalty_null_space(
                [penalty_component_dense_matrix(c, matrix) for c in own], group.size
            )
        built[index] = _raw_construction(null, blocks, group, matrix)
        return built[index]

    return columns


def _raw_construction(null: NDArray, blocks: int, group, matrix) -> ConstructionColumns:
    """A group's construction columns from its penalty null space ``null`` (one block's).

    No penalty, they are the coordinates.  Otherwise the null space is carried to the group's
    raw basis, ``theta = R beta`` (the SSP map ``R_inv``, a factor smooth's ``natural_map``),
    whose coordinates the construction fixes; there it is given an orthonormal basis in a fixed
    order (the projections of the raw-index monomials ``1, t, t^2, ...``, Gram-Schmidt, completed
    by unit vectors), and a block's construction coefficients are ``Q' theta``.  Neither the rows'
    weights, the smoothing parameters nor the whitening enter, so a rotation of the solver's
    coordinates leaves every published constraint where it was.
    """
    width = null.shape[0]
    count = null.shape[1]
    levels = getattr(matrix, "levels", None) if blocks > 1 else None

    def name(block: int, j: int) -> str:
        term = group.name
        if levels is not None and block < len(levels):
            term = f"{group.name}[{levels[block]}]"
        part = "unpenalised part" if count == 1 else f"unpenalised part {j + 1}"
        return f"{term}'s {part}"

    if count == width:
        identity = np.eye(width)
        return ConstructionColumns(local=identity, basis=identity, blocks=blocks, name=name)
    if count == 0:
        return ConstructionColumns(
            local=np.zeros((0, width)), basis=np.zeros((width, 0)), blocks=blocks, name=name
        )
    raw = getattr(matrix, "natural_map" if blocks > 1 else "R_inv", None)
    raw = np.asarray(raw, dtype=np.float64) if raw is not None else None
    if raw is None or raw.ndim != 2 or raw.shape[1] != width:
        raw = np.eye(width)
    spanned = np.linalg.qr(raw @ null)[0]
    length = spanned.shape[0]
    grid = np.linspace(-1.0, 1.0, length)
    probes = [grid**degree for degree in range(length)] + list(np.eye(length))
    ordered: list[NDArray] = []
    for probe in probes:
        vector = spanned @ (spanned.T @ probe)
        for _ in range(2):
            for accepted in ordered:
                vector = vector - accepted * float(accepted @ vector)
        norm = float(np.linalg.norm(vector))
        if norm > SQRT_EPS * float(np.linalg.norm(probe)):
            ordered.append(vector / norm)
        if len(ordered) == count:
            break
    local = np.column_stack(ordered).T @ raw
    basis = null @ np.linalg.inv(local @ null)
    return ConstructionColumns(local=local, basis=basis, blocks=blocks, name=name)


def _dense_base(
    decomposition: RankDecomposition,
    data_diagonal: NDArray,
    rows: int,
    hessian: Callable[[], NDArray],
) -> TruncatedDecomposition:
    """The part of a dense decomposition of the centred Hessian it certified, and the rest.

    The Hessian as formed (``hessian()``) is within ``gamma_(n+2) sqrt(D_ii D_jj)`` entrywise of the
    exact one (``D`` the data Gram, ``n`` its rows; Higham 2002, sec. 3.5): a Jacobi-scaled bound
    ``e`` (``rank._formation_slack``).  The kept part is certified when its smallest equilibrated
    eigenvalue (the retained spectrum less the eigensolver's bar, or a Cholesky factor's ``1 /
    ||L^-1||_F^2`` less its backward error) exceeds ``e`` (Weyl); it inverts ``H`` there, and its
    truncated directions are undecided.  A rank decision that kept an eigenvalue it cannot tell from
    zero is replaced by the recomputed equilibrated spectrum: the eigenvalues past the bound are
    certified, the rest read from the rows.  A factor of the rows states its own floor
    (``_factor_floor``); with no certified part it is vacuous, infinite.
    """
    import scipy.linalg

    from superglm.solvers.rank import _eigensolver_relative_bar, _formation_slack

    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    active = np.flatnonzero(scale > 0.0)
    width = scale.size
    values = decomposition.retained_values
    kept = int(decomposition.rank)

    def certified(floor: float) -> TruncatedDecomposition:
        pseudo = decomposition.pseudo_inverse()
        return TruncatedDecomposition(
            null_vectors=decomposition.null_basis(),
            rank=kept,
            column_scale=scale,
            pre_truncation_condition=float(decomposition.pre_truncation_condition),
            solve=lambda vectors: pseudo @ vectors,
            floor=floor,
        )

    if kept == 0:
        return certified(float("inf"))
    if decomposition.method == "qr_svd":
        # a factor decomposition always records the values it kept
        return certified(_factor_floor(cast(NDArray, values), active.size, rows))
    slack = (
        _formation_slack(
            gamma_up(rows + 2) * np.maximum(np.asarray(data_diagonal)[active], 0.0),
            scale[active] ** 2,
        )
        if active.size
        else 0.0
    )
    if kept < len(decomposition.active_columns) or decomposition.cholesky_factor is None:
        largest = float(np.max(values)) if values is not None and len(values) else active.size
        floor = (
            float(np.min(values)) - _eigensolver_relative_bar(active.size + 1) * largest
            if values is not None and len(values)
            else -float("inf")
        )
    else:
        factor = np.asarray(decomposition.cholesky_factor, dtype=np.float64)
        trtri = scipy.linalg.get_lapack_funcs("trtri", (factor,))
        inverse, info = trtri(factor, lower=1)
        floor = (
            1.0 / float(np.linalg.norm(inverse)) ** 2 - gamma_up(kept + 1) * kept
            if info == 0
            else -float("inf")
        )
    if floor > slack:
        return certified(float(floor - slack))
    formed = hessian()
    equilibrated = formed[np.ix_(active, active)] / np.outer(scale[active], scale[active])
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    spectrum, vectors = np.linalg.eigh(equilibrated)
    bound = slack + _eigensolver_relative_bar(active.size + 1) * float(np.max(np.abs(spectrum)))
    trusted = spectrum > bound
    covariance = np.zeros((width, width))
    lifted = vectors / scale[active, None]
    covariance[np.ix_(active, active)] = (lifted[:, trusted] / spectrum[trusted]) @ lifted[
        :, trusted
    ].T
    undecided = np.zeros((width, int(np.count_nonzero(~trusted))))
    undecided[active] = lifted[:, ~trusted]
    structural = np.setdiff1d(np.arange(width), active)
    if structural.size:
        units = np.zeros((width, structural.size))
        units[structural, np.arange(structural.size)] = 1.0
        undecided = np.hstack((undecided, units))
    condition = float(spectrum[-1] / spectrum[trusted][0]) if np.any(trusted) else float("inf")
    return TruncatedDecomposition(
        null_vectors=undecided,
        rank=int(np.count_nonzero(trusted)),
        column_scale=scale,
        pre_truncation_condition=condition,
        solve=lambda vectors: covariance @ vectors,
        covariance=covariance,
        floor=float(spectrum[trusted][0] - bound) if np.any(trusted) else 0.0,
    )


def _factor_floor(values: NDArray, width: int, rows: int) -> float:
    """A certified lower bound on a row factor's Jacobi-scaled eigenvalues on its kept part.

    ``values`` are the squared singular values the factor kept (``rank.decompose_factor``), of
    the factor with unit columns: the rows' TSQR triangle and the penalty's root, ``k`` columns
    of ``n`` rows.  Forming the rows, their triangle and its SVD are backward stable columnwise,
    within ``gamma_(n k + k^2 + 3)`` of each unit column (Higham 2002, Thm 19.4;
    ``identifiability._row_bound``), so within that times ``sqrt(k)`` in Frobenius norm, and the
    exact rows' singular values are within it of the computed ones (Weyl).  The floor is the
    smallest kept value's root less that bound, squared; zero when nothing is left.
    """
    bound = gamma_up(rows * width + width * width + 3) * math.sqrt(width)
    smallest = math.sqrt(float(np.min(values)))
    return (smallest - bound) ** 2 if smallest > bound else 0.0


def _stepped_residual(model, W: NDArray, rows: Any = None) -> NDArray | None:
    """The working residual a published fit steps its weak directions by, or ``None``.

    The rows' one Newton step is the penalised fit along those directions when the objective is
    quadratic in the coefficients, a Gaussian family with the identity link: its weights do not
    move with the mean.  Under any other family a step from a mode the solver truncated is one
    iteration of a fit, not its optimum, so the record keeps the solver's coefficients along the
    directions (``WeakDirections.fitted``) and the publication resumes penalised IRLS on them
    (``_resumed_weak_fit``).
    """
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink

    if type(model._distribution) is Gaussian and type(model._link) is IdentityLink:
        return _working_residual(model, W, rows)
    return None


def _working_residual(model, W: NDArray, rows: Any = None) -> NDArray:
    """``sqrt(W) (z - eta)`` at the solver's fitted state: the rows' working residual.

    ``z`` is the Fisher working response the fit's last system solved, at its weights ``W``;
    ``rows`` (``W``, ``z - eta``) are the publication's, when it has them.
    """
    weights = np.asarray(W, dtype=np.float64)
    carried = weights > 0.0
    result = np.zeros_like(weights)
    if rows is not None:
        result[carried] = np.sqrt(weights[carried]) * np.asarray(rows[1])[carried]
        return result
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.separation_ops import withheld_weights
    from superglm.solvers.mode_score import linear_predictor
    from superglm.solvers.working_rows import coefficient_working_rows

    solver = model._solver_pirls_result()
    eta = stabilize_eta(linear_predictor(model._dm, solver, model._fit_offset), model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    rows = coefficient_working_rows(
        distribution=model._distribution,
        link=model._link,
        y=np.asarray(model._fit_y_ref, dtype=np.float64),
        mu=mu,
        eta=eta,
        sample_weight=solver_weights(withheld_weights(model), solver),
        prefer_observed=False,
    )
    residual = np.asarray(rows.response, dtype=np.float64) - eta
    result[carried] = np.sqrt(weights[carried]) * residual[carried]
    return result


def _decided_identifiability(
    model,
    decomposition: TruncatedDecomposition,
    active_groups,
    design,
    W,
    *,
    data_diagonal: NDArray,
    data_apply: Callable[[NDArray], NDArray],
    penalty: Callable[[NDArray], tuple[NDArray, NDArray]],
    published=None,
    publish: bool = False,
    residual: NDArray | None = None,
    separated: NDArray | None = None,
) -> Identifiability:
    """The identifiability of a fit whose centred Hessian's decomposition is ``decomposition``.

    ``decomposition`` is in the coordinates of ``active_groups`` (the selected groups),
    ``design`` their rows; ``solvers.identifiability`` decides and explains the record.  A fit
    that ``publish``-es its coefficients gets the gauge on its terms' construction columns and
    its weak directions' covariance, and with ``residual`` (``_working_residual``) their Newton
    step; otherwise the record describes the coefficients as ``published`` (the solver's
    decomposition) holds them.
    """
    original = {group.name: group for group in model._groups}
    columns = np.zeros(0, dtype=np.intp)
    if active_groups:
        columns = np.concatenate(
            [np.arange(original[g.name].start, original[g.name].end) for g in active_groups]
        )
    solver = model._solver_pirls_result()
    return decide_identifiability(
        decomposition,
        columns=columns,
        width=len(solver.beta),
        groups=active_groups,
        design=design,
        W=W,
        data_diagonal=data_diagonal,
        data_apply=data_apply,
        penalty=penalty,
        specs={**getattr(model, "_specs", {}), **getattr(model, "_interaction_specs", {})},
        published=published,
        construction=_construction_columns(model, active_groups) if publish else None,
        residual=residual,
        beta=None if residual is None else np.asarray(solver.beta, dtype=np.float64)[columns],
        support=_prior_support(model, design),
        separated=separated,
    )


def _prior_support(model, design) -> NDArray | None:
    """The rows with positive prior weight, as ``0/1`` weights; ``None`` without the fit's weights.

    Rows a certified separation withholds are off the support (``separation_ops``).
    """
    from superglm.model.separation_ops import withheld_weights

    weights = getattr(model, "_fit_weights", None)
    if weights is None or np.shape(weights) != (design.n,):
        return None
    return (withheld_weights(model, weights) > 0.0).astype(np.float64)


def _structured_penalty_diagonal(penalized) -> NDArray:
    """The diagonal of a structured fit's penalty in its slope coordinates, from its parts.

    An ``fs`` or ``sz`` operator keeps its border and level penalties apart; an ``sz`` level's
    public coordinate also carries the last level's block (``beta_K = -sum_l beta_l``).  A nested
    chain keeps one lambda per node and its border penalty.
    """
    diagonal = np.empty(penalized.shape[0])
    if isinstance(penalized, NestedPenalizedOperator):
        diagonal[penalized.small_indices] = np.diag(penalized.border_penalty)
        diagonal[penalized.structured_indices] = np.concatenate(penalized.node_penalty)
        return diagonal
    local = np.diagonal(penalized.penalty_local, axis1=1, axis2=2)
    if isinstance(penalized, SumToZeroPenalizedOperator):
        local = local[:-1] + local[-1:]
    diagonal[penalized.small_indices] = np.diag(penalized.penalty_small)
    diagonal[penalized.structured_indices] = local
    return diagonal


def _spread_centred_operator(operator: CenteredBlockOperator) -> CenteredBlockOperator:
    """The centred data operator formed at the columns' spread, as ``X_F``'s Gram bound assumes.

    An ``fs`` or ``sz`` leaf system's already is (its ``c0``-shifted moments).  A nested chain's
    retained one centres the raw moments, which cancel for a column far from zero; its leaf
    statistics are taken about the rows' global centre ``c``, so centring them about the weighted
    mean's offset ``m - c`` rounds at the spread (Chan, Golub & LeVeque 1983).
    """
    raw = operator.raw
    if not isinstance(raw, NestedDataOperator):
        return operator
    leaf = dataclasses.replace(raw.leaf, center=np.zeros_like(raw.leaf.center))
    cross = np.array(operator.cross, dtype=np.float64, copy=True)
    cross[raw.small_indices] = leaf.cross.sum(axis=0)
    return CenteredBlockOperator(
        raw=dataclasses.replace(raw, leaf=leaf),
        cross=cross,
        total=operator.total,
        center=cross / operator.total,
    )


def _structured_penalty_product(penalized):
    """``T -> (S T, |S| |T|)`` for a structured fit's penalty, from its parts, in slope coordinates.

    An ``fs`` or ``sz`` operator keeps its border block and one block per level; an ``sz``
    level's public coordinate also carries the last level's (``beta_K = -sum_l beta_l``, so
    ``(S beta)_l = P_l beta_l - P_K beta_K``).  A nested chain keeps one lambda per node and its
    border penalty.
    """
    small = np.asarray(penalized.small_indices, dtype=np.intp)
    levels = np.asarray(penalized.structured_indices, dtype=np.intp)
    if isinstance(penalized, NestedPenalizedOperator):
        border = np.asarray(penalized.border_penalty, dtype=np.float64)
        nodes = np.concatenate(penalized.node_penalty)
        levels = levels.ravel()
    else:
        border = np.asarray(penalized.penalty_small, dtype=np.float64)
        local = np.asarray(penalized.penalty_local, dtype=np.float64)

    def apply(T: NDArray) -> tuple[NDArray, NDArray]:
        applied, magnitude = np.zeros_like(T), np.zeros_like(T)
        applied[small] = border @ T[small]
        magnitude[small] = np.abs(border) @ np.abs(T[small])
        if isinstance(penalized, NestedPenalizedOperator):
            applied[levels] = nodes[:, None] * T[levels]
            magnitude[levels] = np.abs(applied[levels])
            return applied, magnitude
        blocks = T[levels]
        if isinstance(penalized, SumToZeroPenalizedOperator):
            raw = np.concatenate((blocks, -blocks.sum(axis=0)[None]))
            own = np.einsum("lij,ljt->lit", local, raw)
            reach = np.einsum("lij,ljt->lit", np.abs(local[:-1]), np.abs(blocks))
            applied[levels] = own[:-1] - own[-1:]
            # ``|beta_K| <= sum_l |beta_l|``
            magnitude[levels] = reach + (np.abs(local[-1]) @ np.abs(blocks).sum(axis=0))[None]
        else:
            applied[levels] = np.einsum("lij,ljt->lit", local, blocks)
            magnitude[levels] = np.einsum("lij,ljt->lit", np.abs(local), np.abs(blocks))
        return applied, magnitude

    return apply


def _structured_identifiability(
    model,
    state: StructuredLinearSystemState,
    W,
    *,
    publish: bool = False,
    residual: NDArray | None = None,
    separated: NDArray | None = None,
) -> Identifiability:
    """The identifiability of a structured fit, read from its retained factor and moments.

    The factor inverts ``H`` off the directions it truncated (``null_directions``) on pivots it
    certified, so those are the only undecided ones; a factor that truncated none certified ``H``
    nonsingular (``identifiability.certified_identifiability``).  Otherwise the rows are read on
    its truncated directions, the penalty applied from its parts and the data Gram through the
    centred operator, so no ``p x p`` matrix is formed.  The condition the factor certified is
    its border's.  The record is kept on ``state``; ``publish`` and ``residual`` are
    ``_decided_identifiability``'s.
    """
    if state.identifiability is not None:
        return state.identifiability
    record = _structured_record(
        model, state, W, publish=publish, residual=residual, separated=separated
    )
    object.__setattr__(state, "identifiability", record)
    return record


def _structured_record(
    model,
    state: StructuredLinearSystemState,
    W,
    *,
    publish: bool,
    residual: NDArray | None,
    separated: NDArray | None = None,
) -> Identifiability:
    """``_structured_identifiability``'s decision, before it is kept on ``state``."""
    factor = state.profiled_factor
    certified = certified_identifiability(
        TruncatedDecomposition(
            null_vectors=np.zeros((factor.shape[0], 0)),
            rank=int(factor.rank),
            column_scale=np.ones(factor.shape[0]),
            pre_truncation_condition=float(factor.schur_condition_estimate),
        ),
        len(model._solver_pirls_result().beta),
    )
    if certified is not None:
        return certified
    operator = _spread_centred_operator(state.centered_data_operator)
    data_diagonal = compact_operator_diagonal(operator)
    hessian_diagonal = data_diagonal + _structured_penalty_diagonal(state.penalized_operator)
    from superglm.model.separation_ops import separation_boundaries, structured_floor

    decomposition = TruncatedDecomposition(
        null_vectors=factor.null_directions(),
        rank=int(factor.rank),
        column_scale=np.sqrt(np.maximum(hessian_diagonal, 0.0)),
        pre_truncation_condition=float(factor.schur_condition_estimate),
        solve=factor.solve,
        # the existence certificate reads it (``separation_ops.existence_certified``)
        floor=structured_floor(factor, hessian_diagonal) if separation_boundaries(model) else 0.0,
    )
    return _decided_identifiability(
        model,
        decomposition,
        list(model._groups),
        model._dm,
        W,
        data_diagonal=data_diagonal,
        data_apply=operator.matvec,
        penalty=_structured_penalty_product(state.penalized_operator),
        publish=publish,
        residual=residual,
        separated=separated,
    )


def _dense_identifiability(
    model,
    decomposition,
    active_groups,
    design,
    W,
    data_gram,
    *,
    publish=False,
    residual=None,
    separated=None,
) -> Identifiability:
    """``_decided_identifiability`` for a dense decomposition of the centred Hessian.

    The fitted penalty curvature is formed only if the decision reads it: an undecided
    direction's penalty, or a rank decision ``_dense_base`` replaces.
    """

    @functools.cache
    def curvature() -> NDArray:
        return _rank_curvature(model, active_groups)

    return _decided_identifiability(
        model,
        _dense_base(decomposition, np.diag(data_gram), design.n, lambda: data_gram + curvature()),
        active_groups,
        design,
        W,
        data_diagonal=np.diag(data_gram),
        data_apply=lambda vectors: data_gram @ vectors,
        penalty=lambda vectors: (
            curvature() @ vectors,
            np.abs(curvature()) @ np.abs(vectors),
        ),
        published=decomposition,
        publish=publish,
        residual=residual,
        separated=separated,
    )


def _record_identifiability(
    model,
    rank_info,
    design,
    active_groups,
    W,
    data_gram,
    *,
    publish: bool = False,
    residual: NDArray | None = None,
    separated: NDArray | None = None,
) -> Identifiability | None:
    """The record's identifiability, decided when the fit publishes it and kept on the record.

    The fit decides it when it publishes its coefficients (``publish_identified_fit``, with
    ``publish``); a record it did not publish (a constrained fit's face) decides it on first
    inference, describing its coefficients.  ``None`` for a record saved before fits recorded
    it: its standard errors keep their flags (``RankInfo.coefficient_identified``).
    """
    if rank_info.identifiability_policy != "penalised":
        return None
    if rank_info.identifiability is None:
        decided = _dense_identifiability(
            model,
            rank_info.augmented,
            active_groups,
            design,
            W,
            data_gram,
            publish=publish,
            residual=residual,
            separated=separated,
        )
        object.__setattr__(rank_info, "identifiability", decided)
    return rank_info.identifiability


@dataclasses.dataclass
class _PublicationContext:
    """What ``publish_identified_fit`` publishes from: the record and the state it was read on.

    ``certificate`` holds the existence certificate's inputs (``separation_ops.
    existence_certified``): the design, its columns' fitted coordinates, the decomposition, the
    penalty, the data Gram's diagonal and the certified curvature floor.  ``withheld`` is set
    when a certified separation's rows are withheld, with ``separated`` (``[intercept; slopes]``)
    the span the published coefficients keep at the walked fit's values (``walk``) and
    ``explicit`` the separated rows' explicit columns, the span's coordinates there.
    """

    record: Identifiability | None
    state: StructuredLinearSystemState | None
    rank_info: RankInfo | None
    data_gram: NDArray | None
    certificate: dict
    withheld: WithheldFit | None = None
    separated: NDArray | None = None
    walk: Any = None
    explicit: Any = None
    separated_explicit: NDArray | None = None


def _publication_context(
    model, boundaries: tuple[str, ...], *, separated: NDArray | None = None
) -> _PublicationContext | None:
    """The fit's identifiability record, decided at its working weights, and the certificate's inputs.

    ``None`` when the fit publishes nothing: a constrained face, a record saved before fits
    recorded identifiability, a structured factor that certified full rank on a family with no
    response boundary (with one the existence certificate still reads it).
    """
    from superglm.model.separation_ops import structured_floor

    solver = model._solver_pirls_result()
    state = retained_linear_state(model)
    if isinstance(state, StructuredLinearSystemState):
        if state.identifiability is not None:
            return None
        full = int(state.profiled_factor.rank) == state.profiled_factor.shape[0]
        if full and not boundaries:
            return None
        W, rows, _ = _publication_rows(model, boundaries)
        record = _structured_identifiability(
            model,
            state,
            W,
            publish=True,
            residual=_stepped_residual(model, W, rows),
            separated=separated,
        )
        factor = state.profiled_factor
        operator = _spread_centred_operator(state.centered_data_operator)
        data_diagonal = compact_operator_diagonal(operator)
        hessian_diagonal = data_diagonal + _structured_penalty_diagonal(state.penalized_operator)
        floor = (
            structured_floor(factor, hessian_diagonal)
            if full and boundaries
            else record.curvature_floor
        )
        certificate = {
            "design": model._dm,
            "columns": np.arange(factor.shape[0]),
            "decomposition": TruncatedDecomposition(
                null_vectors=factor.null_directions(),
                rank=int(factor.rank),
                column_scale=np.sqrt(np.maximum(hessian_diagonal, 0.0)),
                pre_truncation_condition=float(factor.schur_condition_estimate),
                solve=factor.solve,
            ),
            "penalty": _structured_penalty_product(state.penalized_operator),
            "data_diagonal": data_diagonal,
            "floor": floor,
            "rows": rows,
        }
        return _PublicationContext(record, state, None, None, certificate)
    rank_info = solver.rank_info
    if (
        rank_info is None
        or rank_info.identifiability_policy != "penalised"
        or rank_info.identifiability is not None
        or rank_info.face is not None
    ):
        return None
    W, rows, solved = _publication_rows(model, boundaries, rank_info)
    design, active_groups = _rank_active_state(model, rank_info)
    if design.p == 0:
        return None
    held = rank_info.data_gram
    # the solve's own rows formed its Gram; evaluated rows match it when their bits do
    own = held is not None and (solved or rank_info.data_gram_weights == weights_fingerprint(W))
    data_gram = held if own else _rank_centered_data_gram(design, W)
    record = _record_identifiability(
        model,
        rank_info,
        design,
        active_groups,
        W,
        data_gram,
        publish=True,
        residual=_stepped_residual(model, W, rows),
        separated=separated,
    )

    @functools.cache
    def curvature() -> NDArray:
        return _rank_curvature(model, active_groups)

    certificate_columns = np.asarray(rank_info.selected_columns, dtype=np.intp)
    certificate = {
        "design": design,
        "columns": certificate_columns,
        "decomposition": functools.partial(
            _dense_base,
            rank_info.augmented,
            np.diag(data_gram),
            design.n,
            lambda: data_gram + curvature(),
        ),
        "penalty": lambda vectors: (curvature() @ vectors, np.abs(curvature()) @ np.abs(vectors)),
        "data_diagonal": np.diag(data_gram),
        # the decomposition's floor is the matrix it decomposed: the fit's own Gram, centred
        # where the fit centred it
        "floor": record.curvature_floor if own and record is not None else 0.0,
        "rows": rows,
        "centre": np.asarray(rank_info.mean_x, dtype=np.float64)[certificate_columns]
        if own
        else None,
    }
    return _PublicationContext(record, None, rank_info, data_gram, certificate)


def _publication_rows(
    model, boundaries: tuple[str, ...], rank_info: RankInfo | None = None
) -> tuple[NDArray, Any, bool]:
    """The publication's working weights, its Fisher rows, and whether they are the solve's own.

    One evaluation of the fitted state's rows serves the record and the existence
    certificate, so the certificate reads the very array its record was decided at.  The
    solve's own rows (``RankInfo.take_fitted_rows``: the rows its data Gram was formed at, at
    the coefficients and prior weights published) are read where they hold; otherwise they
    are evaluated (``separation_ops.fisher_rows``, whose weights are
    ``_solver_space_working_weights``'s bit for bit), and without a response boundary only the
    weights are, since no certificate runs.
    """
    from superglm.model.separation_ops import withheld_weights

    solver = model._solver_pirls_result()
    if rank_info is not None:
        held = rank_info.take_fitted_rows(solver, solver_weights(withheld_weights(model), solver))
        if held is not None:
            return held[0], held, True
    if not boundaries:
        return _solver_space_working_weights(model), None, False
    from superglm.model.separation_ops import fisher_rows

    rows = fisher_rows(model, solver, withheld_weights(model))
    return rows[0], rows, False


def _install_record(context: _PublicationContext, record: Identifiability) -> None:
    """Keep ``record`` where readers find it: on the structured state or the rank record."""
    context.record = record
    if context.state is not None:
        object.__setattr__(context.state, "identifiability", record)
    elif context.rank_info is not None:
        object.__setattr__(context.rank_info, "identifiability", record)


def _certified(model, context: _PublicationContext, sides: NDArray, weights: NDArray) -> bool:
    """``separation_ops.existence_certified`` on the context's fit and record.

    The certificate bounds the Hessian on the record's quotient by its null directions, so each
    must be exact lineality, not a near-alias that could separate: one on the explicit columns
    is verified to vanish on every live row exactly (``separation_ops.aliases_exact``).
    """
    from superglm.model.separation_ops import aliases_exact, existence_certified

    record, inputs = context.record, context.certificate
    if record is None:
        return False
    if record.gauge is not None and not aliases_exact(model, record.gauge, sides):
        return False
    decomposition = inputs["decomposition"]
    if callable(decomposition):
        decomposition = decomposition()
    columns = inputs["columns"]
    weak = record.weak
    return existence_certified(
        model,
        solver=model._solver_pirls_result(),
        sides=sides,
        weights=weights,
        decomposition=decomposition,
        columns=columns,
        design=inputs["design"],
        penalty=inputs["penalty"],
        data_diagonal=inputs["data_diagonal"],
        floor=float(inputs["floor"]),
        weak_root=None if weak is None else np.asarray(weak.root)[columns],
        rows=inputs.get("rows"),
        centre=inputs.get("centre"),
    )


def _publish_existence(
    model,
    context: _PublicationContext,
    boundaries: tuple[str, ...],
    *,
    refit: Callable[[Any, NDArray], WithheldFit] | None = None,
) -> _PublicationContext:
    """Decide whether the fit's estimate exists, and publish the fit that decision gives.

    The fit's score proves existence where it can (``separation_ops.existence_certified``);
    otherwise the rows decide (``separation_ops.decide_separation``).  A direction verified exactly
    is published as the fit on the rows it keeps (``_withheld_publication``) once that fit's own
    score proves those rows separate no further; short of that it is a proven lower bound
    (``"partial"``), and the walked fit is published with the decision on its record.  A fit refused
    under ``"error"`` (``refuses_on_publication``) publishes nothing, so no withheld fit is run.
    The decision is kept on the record whatever it is, ``"absent"`` included, so a reader tells a
    decided absence from one never built.  ``refit`` forms the fit on the kept rows
    (``separation_ops.withheld_refit`` by default).
    """
    from superglm.diagnostics.separation import refuses_on_publication, response_sides
    from superglm.model.separation_ops import (
        coefficient_names,
        decide_separation,
        separated_labels,
    )

    model._separated_rows = None
    weights = np.asarray(model._fit_weights, dtype=np.float64)
    sides = response_sides(model._fit_y_ref, weights, boundaries)
    certified = _certified(model, context, sides, weights)
    decision, strict, explicit, coordinates = decide_separation(
        model, sides=sides, boundaries=boundaries, certified=certified
    )
    if strict is not None and strict.size == np.count_nonzero(sides != 2):
        # complete separation: every row is separated and none is kept, so no data bound any
        # coefficient the penalty leaves free; the walked fit is published as it stopped
        span = _complete_span(model, context, len(model._solver_pirls_result().beta))
        labels, labelled = separated_labels(model, strict, sides, coordinates)
        decision = dataclasses.replace(
            decision,
            state="present",
            decided_by="linear program: complete separation",
            span=span,
            names=coefficient_names(model, span),
            labels=labels,
            labelled_rows=labelled,
            complete=True,
        )
    elif strict is not None and not refuses_on_publication(model):
        published = _withheld_publication(
            model, context, decision, strict, explicit, coordinates, sides, boundaries, refit
        )
        if published is not None:
            return published
    if decision.state == "unresolved":
        if _interior_full_rank(model, context, sides):
            decision = dataclasses.replace(decision, state="absent", decided_by="interior rank")
    if context.record is not None:
        _install_record(context, dataclasses.replace(context.record, separation=decision))
    return context


def _complete_span(model, context: _PublicationContext, width: int) -> NDArray:
    """The directions a complete separation drives to infinity: the penalty's null space.

    Every positive-prior row is separated, so a separating direction is strict on all of them
    and every direction of ``null(S)`` near it separates too: the directions of recession are
    open in ``null(S)`` and span it, with the intercept (the w5m2 gate review, sec. 3).  A
    direction the penalty curves is not among them.  ``null(S)`` is read from the fitted
    penalty (``_rank_curvature``): eigenvalues within ``gamma_p p`` of its largest, and the
    coordinates its basis reaches past the square root of eps.  A structured fit, whose
    penalty is held as an operator, reports every coordinate.  ``[intercept; slopes]``.
    """
    from superglm._roundoff import gamma

    if context.rank_info is None:
        return np.eye(width + 1)
    _, active_groups = _rank_active_state(model, context.rank_info)
    columns = np.asarray(context.rank_info.selected_columns, dtype=np.intp)
    curvature = _rank_curvature(model, active_groups)
    size = curvature.shape[0]
    values, vectors = np.linalg.eigh(curvature) if size else (np.zeros(0), np.zeros((0, 0)))
    largest = float(np.max(np.abs(values), initial=0.0))
    null = vectors[:, values <= gamma(max(size, 1)) * size * largest]
    reach = np.max(np.abs(null), axis=1, initial=0.0) if null.size else np.zeros(size)
    reached = columns[reach > SQRT_EPS * float(np.max(reach, initial=0.0))]
    span = np.zeros((width + 1, 1 + reached.size))
    span[0, 0] = 1.0
    span[1 + reached, 1 + np.arange(reached.size)] = 1.0
    return span


def _interior_full_rank(model, context: _PublicationContext, sides: NDArray) -> bool:
    """Whether the interior rows alone identify every coefficient: then no direction separates.

    A direction of recession is zero on every interior row and in the penalty's null space, so
    when the Hessian of the interior rows, ``X_I'X_I + S`` at unit weight, is nonsingular there
    is none (Santos Silva & Tenreyro 2010; Correia, Guimaraes & Zylkin 2019, sec. 3.2).  The
    decision is the owner's two-step rule on that Hessian (``_dense_base``'s formation bound,
    then the rows on what it leaves undecided, ``decide_identifiability``): certified when no
    direction is null and none unresolved.  Dense fits; a structured fit leaves it unresolved.
    """
    if context.rank_info is None:
        return False
    interior = ((sides == 0) & (np.asarray(model._fit_weights) > 0.0)).astype(np.float64)
    if not np.any(interior):
        return False
    design, active_groups = _rank_active_state(model, context.rank_info)
    curvature = _rank_curvature(model, active_groups)
    gram = _rank_centered_data_gram(design, interior)
    hessian = gram + curvature
    decomposition = decompose_gram_if_authoritative(hessian, psd_by_construction=True)
    if decomposition is None:
        decomposition = decompose_gram(hessian, allow_indefinite=True)
    diagonal = np.diag(gram)
    record = decide_identifiability(
        _dense_base(decomposition, diagonal, design.n, lambda: hessian),
        columns=np.asarray(context.rank_info.selected_columns, dtype=np.intp),
        width=len(model._solver_pirls_result().beta),
        groups=active_groups,
        design=design,
        W=interior,
        data_diagonal=diagonal,
        data_apply=lambda vectors: gram @ vectors,
        penalty=lambda vectors: (curvature @ vectors, np.abs(curvature) @ np.abs(vectors)),
        support=interior,
    )
    # a "fit" alias is a coordinate the decomposition truncated and the rows identify: not null
    return record.curvature_floor > 0.0 and all(alias.source == "fit" for alias in record.aliases)


def _withheld_publication(
    model,
    context: _PublicationContext,
    decision,
    strict: NDArray,
    explicit,
    coordinates: NDArray,
    sides: NDArray,
    boundaries: tuple[str, ...],
    refit: Callable[[Any, NDArray], WithheldFit] | None = None,
) -> _PublicationContext | None:
    """The fit with the separated rows withheld, if it certifies them as every separated row.

    The withheld fit (``separation_ops.withheld_refit``) is installed, its record decided on the
    rows it keeps, and its null directions written exactly on the explicit columns
    (``separation_ops.separated_span``): those nonzero on a withheld row are the separated span,
    held where the walk left them.  Its own score must prove existence on the kept rows
    (maximality: every separated row is withheld); otherwise the walked fit is restored and
    ``None`` returned.
    """
    from superglm.diagnostics.separation import Separation
    from superglm.model.separation_ops import (
        coefficient_names,
        existence_certified,
        lift_span,
        separated_labels,
        separated_span,
        withheld_refit,
        withheld_weights,
    )

    walk = model._solver_pirls_result()
    saved = (
        model._result,
        model._solver_result,
        getattr(model, "_linear_system_state", None),
        getattr(getattr(model, "_reml_result", None), "pirls_result", None),
    )

    def restore() -> None:
        model._result, model._solver_result = saved[0], saved[1]
        model._linear_system_state = saved[2]
        reml = getattr(model, "_reml_result", None)
        if reml is not None:
            reml.pirls_result = saved[3]
        model._separated_rows = None

    try:
        fit = (withheld_refit if refit is None else refit)(model, strict)
    except Exception:  # pragma: no cover - a refit the solver refuses leaves it unresolved
        return None
    # the published fit is the fit the caller ran, its coefficients on the kept rows refined:
    # its iterations, stop and convergence are the walk's (and the refit's convergence)
    fit.result = dataclasses.replace(
        fit.result,
        n_iter=walk.n_iter,
        converged=bool(walk.converged and fit.result.converged),
        iteration_log=walk.iteration_log,
        termination_reason=walk.termination_reason,
        mean_space_boundary_rows=walk.mean_space_boundary_rows,
        mean_space_boundary_unresolved=walk.mean_space_boundary_unresolved,
    )
    model._separated_rows = strict
    model._result = model._solver_result = fit.result
    if fit.state is not None:
        model._linear_system_state = fit.state
    reml = getattr(model, "_reml_result", None)
    if reml is not None:
        reml.pirls_result = fit.result
    width = len(walk.beta)
    live = np.flatnonzero(sides != 2)
    kept = np.setdiff1d(live, strict)
    first = _publication_context(model, boundaries)
    gauge = None if first is None or first.record is None else first.record.gauge
    if first is None or gauge is None:
        restore()
        return None
    null = np.vstack((np.asarray(gauge.intercept)[None, :], np.asarray(gauge.null)))
    spans = separated_span(explicit, coordinates, null, strict, kept)
    if spans is None or spans[1].shape[1] == 0:
        restore()
        return None
    separated = lift_span(spans[1], coordinates, width)
    # the record again, with the separated span held where the walk left it
    if first.state is not None:
        object.__setattr__(first.state, "identifiability", None)
    elif first.rank_info is not None:
        object.__setattr__(first.rank_info, "identifiability", None)
    context_r = _publication_context(model, boundaries, separated=separated[1:])
    if context_r is None or context_r.record is None:
        restore()
        return None
    sides_r = sides.copy()
    sides_r[strict] = 2
    inputs = context_r.certificate
    decomposition = inputs["decomposition"]
    if callable(decomposition):
        decomposition = decomposition()
    weak = context_r.record.weak
    if not existence_certified(
        model,
        solver=fit.result,
        sides=sides_r,
        weights=withheld_weights(model),
        decomposition=decomposition,
        columns=inputs["columns"],
        design=inputs["design"],
        penalty=inputs["penalty"],
        data_diagonal=inputs["data_diagonal"],
        floor=float(inputs["floor"]),
        weak_root=None if weak is None else np.asarray(weak.root)[inputs["columns"]],
        rows=inputs.get("rows"),
        centre=inputs.get("centre"),
    ):
        restore()
        return None
    labels, labelled = separated_labels(model, strict, sides, coordinates)
    separation = Separation(
        "present",
        "linear program",
        rows=np.asarray(strict, dtype=np.intp),
        span=separated,
        boundaries=boundaries,
        unique=decision.unique,
        labels=labels,
        labelled_rows=labelled,
        names=coefficient_names(model, separated),
        work=decision.work,
    )
    _install_record(context_r, dataclasses.replace(context_r.record, separation=separation))
    context_r.withheld = fit
    context_r.separated = separated
    context_r.walk = walk
    context_r.explicit = explicit[strict]
    context_r.separated_explicit = spans[1]
    return context_r


# The resumed fit's budget: its iterations and the step halvings each may take.
_RESUME_ITERATIONS = 50
_RESUME_HALVINGS = 30
# The resumed fit stops when its Newton decrement is under this fraction of the objective.
_RESUME_TOLERANCE = 1e-13


def _resumed_weak_fit(model, context: _PublicationContext) -> _PublicationContext | None:
    """D6 under any family: penalised IRLS resumed on the identified subspace, at the fit's lambda.

    The solver truncated directions ``T`` the rows identify (``WeakDirections``).  Each iteration
    rebuilds the centred system, its decomposition and the owner's row read at the current Fisher
    weights, and takes the joint Newton step: ``V g`` on the certified part, ``U C^-1 U'g`` on the
    weak directions (``U = T - V H T`` carries their coupling), and the intercept's own.  A step is
    halved until ``D + beta'S beta`` does not rise, at most ``_RESUME_HALVINGS`` times; it stops at
    stationarity (``g' H^+ g`` under the fit's tolerance) or after ``_RESUME_ITERATIONS``.  The last
    accepted state's record, covariance and EDF are published, the rank record rebuilt at its
    weights; ``None`` (6a's coefficients kept, the summary saying so) when no step is accepted or
    stationarity is not reached.  Dense fits only.
    """
    from superglm.model.separation_ops import fisher_rows, withheld_weights
    from superglm.solvers.centered_system import build_centered_system

    record, rank_info = context.record, context.rank_info
    if record is None or rank_info is None or record.weak is None or record.weak.fitted:
        return None
    solver = model._solver_pirls_result()
    design, active_groups = _rank_active_state(model, rank_info)
    columns = np.asarray(rank_info.selected_columns, dtype=np.intp)
    curvature = _rank_curvature(model, active_groups)
    weights = withheld_weights(model)
    # stationarity tighter than any fit's tolerance: Newton converges quadratically there, so
    # it costs an iteration, and a looser stop would leave the weak directions short of the mode
    tol = min(float(getattr(model, "_tol", 1e-8)), _RESUME_TOLERANCE)
    width = len(solver.beta)

    def raw(beta: NDArray, intercept: float):
        return dataclasses.replace(
            solver, beta=beta, intercept=intercept, centred_intercept=None, state_center=None
        )

    def objective(state) -> float:
        from superglm._utils import weighted_row_sum
        from superglm.distributions import clip_mu
        from superglm.links import stabilize_eta
        from superglm.solvers.mode_score import linear_predictor

        eta = stabilize_eta(linear_predictor(model._dm, state, model._fit_offset), model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)
        deviance = weighted_row_sum(
            solver_weights(weights, state),
            model._distribution.deviance_unit,
            np.asarray(model._fit_y_ref, dtype=np.float64),
            mu,
        )
        slopes = np.asarray(state.beta, dtype=np.float64)[columns]
        return float(deviance + slopes @ curvature @ slopes)

    def decide(state):
        W, residual = fisher_rows(model, state, weights)
        system = build_centered_system(dm=design, W=W, z_off=np.zeros(design.n), penalty=curvature)
        decomposition = decompose_gram_if_authoritative(system.hessian, psd_by_construction=True)
        if decomposition is None:
            decomposition = decompose_gram(system.hessian, allow_indefinite=True)
        diagonal = np.diag(system.data_gram)
        truncated = _dense_base(decomposition, diagonal, design.n, lambda: system.hessian)
        carried = W > 0.0
        stepped = np.zeros_like(W)
        stepped[carried] = np.sqrt(W[carried]) * residual[carried]
        decided = decide_identifiability(
            truncated,
            columns=columns,
            width=width,
            groups=active_groups,
            design=design,
            W=W,
            data_diagonal=diagonal,
            data_apply=lambda vectors: system.data_gram @ vectors,
            penalty=lambda vectors: (curvature @ vectors, np.abs(curvature) @ np.abs(vectors)),
            specs={**getattr(model, "_specs", {}), **getattr(model, "_interaction_specs", {})},
            construction=_construction_columns(model, active_groups),
            residual=stepped,
            beta=np.asarray(state.beta, dtype=np.float64)[columns],
            support=_prior_support(model, design),
        )
        r = W * residual
        rsum = math.fsum(r.tolist())
        slopes = np.asarray(state.beta, dtype=np.float64)[columns]
        gradient = (
            np.asarray(design.rmatvec(r), dtype=np.float64)
            - system.mean_x * rsum
            - curvature @ slopes
        )
        step = np.asarray(cast(Callable, truncated.solve)(gradient), dtype=np.float64)
        if decided.weak is not None:
            root = np.asarray(decided.weak.root)[columns]
            step = step + root @ (root.T @ gradient)
        intercept_step = rsum / system.sum_w - float(system.mean_x @ step)
        decrement = float(gradient @ step + rsum * intercept_step)
        return system, decomposition, decided, step, intercept_step, decrement

    beta = np.asarray(solver.beta, dtype=np.float64).copy()
    intercept = float(solver.intercept)
    state = raw(beta, intercept)
    current = objective(state)
    stationary = False
    for _ in range(_RESUME_ITERATIONS):
        system, decomposition, decided, step, intercept_step, decrement = decide(state)
        if not (math.isfinite(decrement) and math.isfinite(current)):
            return None
        if decrement <= tol * (abs(current) + tol):
            stationary = True
            break
        accepted = False
        scale = 1.0
        for _ in range(_RESUME_HALVINGS):
            moved = beta.copy()
            moved[columns] += scale * step
            trial = raw(moved, intercept + scale * intercept_step)
            value = objective(trial)
            if math.isfinite(value) and value <= current + 1e-12 * abs(current):
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return None
        beta, intercept, state, current = moved, intercept + scale * intercept_step, trial, value
    if not stationary:
        return None
    feature_edf = np.zeros(len(rank_info.mean_x))
    inverse = (
        decided.covariance if decided.covariance is not None else decomposition.pseudo_inverse()
    )
    feature_edf[columns] = np.diag(inverse @ system.data_gram)
    M_beta = system.hessian + system.sum_w * np.outer(system.mean_x, system.mean_x)
    coefficient = decompose_gram_if_authoritative(M_beta, psd_by_construction=True)
    final_rows = fisher_rows(model, state, weights)
    W_final = final_rows[0]
    resumed_rank = dataclasses.replace(
        rank_info,
        augmented=decomposition,
        coefficient=coefficient if coefficient is not None else decompose_gram(M_beta),
        sum_w=float(system.sum_w),
        mean_x=_spread(system.mean_x, columns, len(rank_info.mean_x)),
        data_gram=system.data_gram,
        data_gram_weights=weights_fingerprint(W_final),
        feature_edf=feature_edf,
        feature_edf1=None,
        group_edf={
            group.name: float(np.sum(feature_edf[columns][group.sl])) for group in active_groups
        },
        identifiability=decided,
    )
    resumed = dataclasses.replace(
        solver,
        beta=beta,
        intercept=intercept,
        centred_intercept=None,
        state_center=None,
        rank_info=resumed_rank,
        effective_df=1.0 + float(np.sum(feature_edf)),
    )
    if model._result is solver:
        model._result = resumed
    reml = getattr(model, "_reml_result", None)
    if reml is not None and getattr(reml, "pirls_result", None) is solver:
        reml.pirls_result = resumed
    model._solver_result = resumed
    certificate = dict(
        context.certificate,
        decomposition=_dense_base(
            decomposition, np.diag(system.data_gram), design.n, lambda: system.hessian
        ),
        data_diagonal=np.diag(system.data_gram),
        floor=decided.curvature_floor,
        rows=final_rows,
        centre=np.asarray(system.mean_x, dtype=np.float64),
    )
    return dataclasses.replace(
        context,
        record=decided,
        rank_info=resumed_rank,
        data_gram=system.data_gram,
        certificate=certificate,
    )


def _reml_pirls(model):
    reml = getattr(model, "_reml_result", None)
    return None if reml is None else getattr(reml, "pirls_result", None)


def _set_reml_pirls(model, result) -> None:
    reml = getattr(model, "_reml_result", None)
    if reml is not None:
        reml.pirls_result = result


def _spread(values: NDArray, columns: NDArray, width: int) -> NDArray:
    """``values`` on ``columns`` of a ``width`` vector, zero elsewhere."""
    spread = np.zeros(width)
    spread[columns] = values
    return spread


def _restore_separated(
    model, solver, beta: NDArray, intercept: float, context: _PublicationContext
) -> tuple[NDArray, float]:
    """The withheld fit's coefficients moved along the separated span to the walk's predictor.

    The span is exactly null on the kept rows, so no kept fitted value moves; on the separated rows
    the move is the least-squares match of the walked fit's predictor (the walk stopped near the
    boundary, where the estimate's limit puts those rows).  Each separated row's residual is read
    relative to its own target, floored at the fit's predictor scale, so no right-hand side exceeds
    two: unweighted, a row the walk drove to ``-1e27`` sets a normwise solver's error to ``u 1e27``
    on every component, the shared one a kept row reads too.  Columns are then equilibrated (van der
    Sluis, Numer. Math. 14, 1969), so each component's error is relative to its own column's scale;
    with every target within the floor the solution is the unweighted one.
    """
    from superglm.solvers.mode_score import linear_predictor

    walk, separated = context.walk, cast(NDArray, context.separated)
    rows = np.asarray(cast("WithheldFit", context.withheld).rows, dtype=np.intp)
    target = np.asarray(linear_predictor(model._dm, walk, model._fit_offset), dtype=np.float64)
    provisional = dataclasses.replace(
        solver,
        beta=beta,
        intercept=intercept,
        centred_intercept=None,
        state_center=None,
    )
    current = np.asarray(linear_predictor(model._dm, provisional, model._fit_offset), np.float64)
    moves = np.asarray(context.explicit @ context.separated_explicit, dtype=np.float64)
    floor = max(1.0, float(np.max(np.abs(current))))
    weight = 1.0 / np.maximum(np.abs(target[rows]), floor)
    weighted = moves * weight[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    scale = 1.0 / np.where(norms > 0.0, norms, 1.0)
    scaled = np.linalg.lstsq(weighted * scale, (target - current)[rows] * weight, rcond=None)[0]
    step = scaled * scale
    return beta + separated[1:] @ step, float(intercept + separated[0] @ step)


def release_fitted_rows(model) -> None:
    """Drop the Fisher rows the model's solves left on their records (``RankInfo.fitted_rows``).

    Their reader is the publication, which has run; a publication that read none (a
    constrained face, a selection penalty, a SCOP fit) leaves them, and they are not kept.
    """
    reml = getattr(model, "_reml_result", None)
    for result in (
        getattr(model, "_solver_result", None),
        getattr(model, "_result", None),
        getattr(reml, "pirls_result", None),
    ):
        rank_info = getattr(result, "rank_info", None)
        if isinstance(rank_info, RankInfo):
            rank_info.release_fitted_rows()


# ── The identifiability record's lifecycle ──────────────────────────────────

RecordTag = Literal["saved", "unfitted", "decided", "stale", "legacy", "frozen"]

#: The identifiability record's lifecycle: for each tag, the events it takes and the tag each
#: leads to; ``move_record`` is its one writer.  ``"decided"``: the record the fit's publication
#: decided at its coefficients (``publish_identified_fit``; a fit that publishes none, such as
#: a constrained or selection-penalty one, forms its inference at the same coefficients on first
#: read).  ``"stale"``: the coefficients were revised after it (an editor edit, Final fit's
#: carry, a shape repair: ``fit_state.publish_revised_coefficients``), so its inference is the
#: original fit's.  A smoothing parameter of a fitted model changes only by a refit, which
#: publishes.  ``"saved"`` is a fitted model that holds no tag: a fit not yet published, or a
#: pickle from before the record had a lifecycle, whose move is its load
#: (``saved_record.migrate_saved_record``): ``"legacy"`` when it kept its fit state (its saved
#: inference is dropped, and its first inference decides the record at the saved coefficients,
#: ``"decide saved"``), ``"frozen"`` when it is compact (it keeps the inference it was saved
#: with), ``"stale"`` when it was edited.  A pickle saved since keeps its tag.
RECORD_MOVES: Mapping[str, Mapping[str, str]] = {
    "saved": {
        "publish": "decided",
        "revise": "stale",
        "load warm": "legacy",
        "load compact": "frozen",
        "load stale": "stale",
    },
    "unfitted": {"publish": "decided"},
    "decided": {"publish": "decided", "revise": "stale"},
    "stale": {"publish": "decided", "revise": "stale"},
    "legacy": {"publish": "decided", "revise": "stale", "decide saved": "decided"},
    "frozen": {"publish": "decided", "revise": "stale"},
}


def record_tag(model) -> RecordTag:
    """The model's place in ``RECORD_MOVES``; a model holding no tag is ``"saved"`` once fitted."""
    tag = model.__dict__.get("_record_tag")
    if tag is not None:
        return tag
    return "unfitted" if model.__dict__.get("_result") is None else "saved"


def move_record(model, event: str) -> RecordTag:
    """Take ``event`` from the model's tag to the one ``RECORD_MOVES`` gives, and return it."""
    before = record_tag(model)
    after = RECORD_MOVES[before].get(event)
    if after is None:
        raise RuntimeError(f"identifiability record: {before} --{event}--> is not a move")
    model.__dict__["_record_tag"] = after
    return cast(RecordTag, after)


def decide_saved_record(model) -> None:
    """A ``"legacy"`` model's record, decided at its saved coefficients on its first inference."""
    if model.__dict__.get("_record_tag") != "legacy":
        return
    from superglm.model.saved_record import decide_at_saved_coefficients

    decide_at_saved_coefficients(model)
    move_record(model, "decide saved")


def publish_identified_fit(model) -> None:
    """Publish the fit its identifiability record decides (``solvers.identifiability``).

    The record is decided now rather than on first inference, at the fit's working weights.  Its
    weak directions, which the solver truncated and the penalised fit identifies, are fitted by
    the one Newton step their rows give (``WeakDirections``): their coefficients, fitted values,
    EDF and scale move, as the penalised fit has them (mgcv's, not ``lm``'s).  Then the
    coefficients move along ``null(H)`` onto the record's constraints (``Gauge``): no fitted
    value moves there, the solver intercept takes each direction's intercept component and a
    centred state's ``alpha`` its offset from the state's centre.  The EDF is the published
    fit's (``identifiability.published_edf``).  A constrained fit and a record saved before fits
    recorded identifiability are left alone; a structured factor that truncated nothing
    certified full rank.  A selection penalty that can zero a group is not quadratic: moving
    along a direction its local curvature leaves null changes its value, so such a fit keeps
    its solver's coefficients and its record describes them.
    """
    from superglm.penalties.base import penalty_can_zero_groups

    move_record(model, "publish")
    solver = getattr(model, "_solver_result", None)
    if solver is None or model._dm is None or getattr(solver, "scop_inference", None) is not None:
        return
    if penalty_can_zero_groups(fitted_penalty(model)):
        return
    from superglm.model.separation_ops import separation_boundaries

    boundaries = separation_boundaries(model)
    context = _publication_context(model, boundaries)
    if context is None:
        return
    # D6: the weak directions are fitted first, by the resumed IRLS (bounded: a direction whose
    # estimate diverges does not reach stationarity and is left as 6a left it); the existence
    # decision is then taken at the state published, and a resumption it cannot certify finite
    # is undone
    walked = (model._result, model._solver_result, _reml_pirls(model))
    resumed = _resumed_weak_fit(model, context)
    if boundaries:
        decided = _publish_existence(model, context if resumed is None else resumed, boundaries)
        separation = None if decided.record is None else decided.record.separation
        if (
            resumed is not None
            and separation is not None
            and separation.state
            in (
                "partial",
                "unresolved",
            )
        ):
            model._result, model._solver_result = walked[0], walked[1]
            _set_reml_pirls(model, walked[2])
            _install_record(
                context,
                dataclasses.replace(cast(Identifiability, context.record), separation=separation),
            )
        else:
            context = decided
    elif resumed is not None:
        context = resumed
    solver = model._solver_pirls_result()
    record, state, rank_info, data_gram = (
        context.record,
        context.state,
        context.rank_info,
        context.data_gram,
    )
    if record is None or (
        record.gauge is None
        and record.weak is None
        and record.covariance is None
        and context.withheld is None
    ):
        return
    # The EDF moves where the published fit's information differs from its solver's: weak
    # directions fitted, or a decomposition that kept a direction of ``null(H)`` replaced.  A
    # gauge alone moves along ``null(H)``, where ``D N = 0``, and leaves the EDF the solver's.
    refit_edf = record.weak is not None or record.covariance is not None
    edf_total = None
    if refit_edf and isinstance(state, StructuredLinearSystemState):
        factor, operator = state.profiled_factor, state.centered_data_operator
        base = factor.inverse_operator_diagonal(operator)
        if _hat_trace_ledger(solver, base):
            edf, edf1 = published_edf(
                base,
                factor.inverse_operator_square_diagonal(operator),
                factor.solve,
                operator.matvec,
                record.weak,
                np.arange(factor.shape[0]),
            )
            record = dataclasses.replace(record, edf=edf, edf1=edf1)
            object.__setattr__(state, "identifiability", record)
            edf_total = 1.0 + float(np.sum(edf))
    elif refit_edf and rank_info is not None:
        columns = np.asarray(rank_info.selected_columns, dtype=np.intp)
        _, active_groups = _rank_active_state(model, rank_info)
        inverse = (
            record.covariance
            if record.covariance is not None
            else rank_info.augmented.pseudo_inverse()
        )
        influence = inverse @ data_gram
        if _hat_trace_ledger(solver, rank_info.feature_edf[columns]):
            selected_edf, selected_edf1 = published_edf(
                np.diag(influence).copy(),
                np.einsum("ij,ji->i", influence, influence),
                lambda vectors: inverse @ vectors,
                lambda vectors: data_gram @ vectors,
                record.weak,
                columns,
            )
            edf, edf1 = np.zeros(len(rank_info.mean_x)), np.zeros(len(rank_info.mean_x))
            edf[columns], edf1[columns] = selected_edf, selected_edf1
            edf[np.abs(edf) < 100.0 * np.finfo(float).eps] = 0.0
            for values in (edf, edf1):
                values.setflags(write=False)
            record = dataclasses.replace(record, edf=edf, edf1=edf1)
            rank_info = dataclasses.replace(
                rank_info,
                feature_edf=edf,
                feature_edf1=edf1,
                group_edf={
                    group.name: float(np.sum(selected_edf[group.sl])) for group in active_groups
                },
            )
            edf_total = float(rank_info.intercept_edf) + float(np.sum(edf))
    if rank_info is not None:
        rank_info = dataclasses.replace(rank_info, identifiability=record)
    beta = np.asarray(solver.beta, dtype=np.float64)
    intercept = float(solver.intercept)
    moved, moved_intercept = beta, intercept
    if record.weak is not None:
        moved = beta + record.weak.step
        moved_intercept = intercept + record.weak.intercept_step
    if record.gauge is not None:
        moved, moved_intercept, _ = record.gauge.coefficients(moved, moved_intercept)
    if context.withheld is not None:
        moved, moved_intercept = _restore_separated(model, solver, moved, moved_intercept, context)
    changes: dict = {"beta": moved, "intercept": moved_intercept}
    if edf_total is not None:
        changes["effective_df"] = edf_total
    alpha, centre = solver.centred_intercept, solver.state_center
    if alpha is not None and centre is not None:
        # ``b0 = alpha - c_s' beta``: a move ``m`` with ``b0`` moving by ``d`` moves ``alpha``
        # by ``c_s' m + d``
        changes["centred_intercept"] = (
            float(alpha)
            + float(np.asarray(centre, dtype=np.float64) @ (moved - beta))
            + (moved_intercept - intercept)
        )
    if rank_info is not None:
        changes["rank_info"] = rank_info
    if solver.reml_hessian_rank is not None and record.rank is not None:
        # the intercept and every slope direction the published fit identifies
        changes["reml_hessian_rank"] = 1 + record.rank
    published = dataclasses.replace(solver, **changes)
    withheld = context.withheld is not None
    refit = record.weak is not None or published.effective_df != solver.effective_df or withheld
    if record.weak is not None or withheld:
        published = dataclasses.replace(published, deviance=_published_deviance(model, published))
    if model._result is solver:
        model._result = published
    reml = getattr(model, "_reml_result", None)
    if reml is not None and getattr(reml, "pirls_result", None) is solver:
        reml.pirls_result = published
    model._solver_result = published
    if refit:
        _refresh_published_scale(model)


def _hat_trace_ledger(solver, edf: NDArray) -> bool:
    """Whether the fit's EDF is its influence trace, ``1 + sum(diag(F))``, which the record can redo.

    A selection penalty's df allocation (Breheny & Huang 2009) is another ledger, and the
    published fit keeps it.
    """
    total = 1.0 + float(np.sum(edf))
    return abs(float(solver.effective_df) - total) <= 1e-9 * max(1.0, abs(total))


def _published_deviance(model, result) -> float:
    """The deviance of ``result``'s fitted values, in physical units (as the fit reports it)."""
    from superglm._utils import weighted_row_sum
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.solvers.mode_score import linear_predictor

    eta = stabilize_eta(linear_predictor(model._dm, result, model._fit_offset), model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    weights = solver_weights(model._fit_weights, result)
    deviance = weighted_row_sum(
        weights,
        model._distribution.deviance_unit,
        np.asarray(model._fit_y_ref, dtype=np.float64),
        mu,
    )
    return float(np.ldexp(deviance, -int(result.weight_exponent)))


def _refresh_published_scale(model) -> None:
    """The fitted means, scale and fit statistics of a fit whose published values moved.

    The scale is the one its route publishes: REML's profile at the penalised mode for a REML
    fit of a Gaussian or Gamma family (``reml_finalize.compute_profiled_phi``), else the Pearson
    statistic over the residual degrees of freedom; a known scale stays one.  Either is the
    kept rows' (``separation_ops.withheld_weights``): a row a certified separation withholds is
    predicted where the walk stopped, so its residual is the walk's, not the fit's.
    """
    from superglm.distributions import Gamma, Gaussian, clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.fit_ops import _compute_fit_stats
    from superglm.model.separation_ops import withheld_weights
    from superglm.solvers.dispersion import (
        model_weight_semantics,
        pearson_residual_degrees_of_freedom,
    )
    from superglm.solvers.mode_score import linear_predictor
    from superglm.solvers.working_rows import pearson_chi2

    result = model._solver_result
    eta = stabilize_eta(linear_predictor(model._dm, result, model._fit_offset), model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    y = np.asarray(model._fit_y_ref, dtype=np.float64)
    weights = np.asarray(model._fit_weights, dtype=np.float64)
    kept = withheld_weights(model, weights)
    solver_phi = None
    if getattr(model._distribution, "scale_known", True):
        phi = 1.0
    elif getattr(model, "_reml_lambdas", None) is not None and isinstance(
        model._distribution, Gaussian | Gamma
    ):
        from superglm.model.reml_finalize import compute_profiled_phi

        phi = compute_profiled_phi(
            model,
            y=y,
            sample_weight=kept,
            lambdas=model._reml_lambdas,
            reml_penalties=model._reml_penalties,
            pirls_result=result,
            boundary_phi=result.phi,
        )
    else:
        scaled = solver_weights(kept, result)
        pearson = pearson_chi2(distribution=model._distribution, y=y, mu=mu, sample_weight=scaled)
        phi = pearson / pearson_residual_degrees_of_freedom(
            kept, result.effective_df, weight_semantics=model_weight_semantics(model)
        )
        if result.weight_exponent:
            solver_phi = phi
            phi = float(np.ldexp(phi, -int(result.weight_exponent)))
    refreshed = dataclasses.replace(result, phi=float(phi), solver_phi=solver_phi)
    if model._result is result:
        model._result = refreshed
    reml = getattr(model, "_reml_result", None)
    if reml is not None and getattr(reml, "pirls_result", None) is result:
        reml.pirls_result = refreshed
    model._solver_result = refreshed
    result = refreshed
    model._fit_mu = mu
    model._fit_stats = _compute_fit_stats(
        y,
        mu,
        weights,
        model._fit_offset,
        model._distribution,
        model._link,
        float(phi),
        null_mu=model._fit_null_mu,
        weight_semantics=model_weight_semantics(model),
        solver_result=result,
    )
    model._summary_cache = None


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
    """Build lazy covariance views over one retained structured fit.

    A fit published with an identifiability gauge or weak directions
    (``publish_identified_fit``) reads its covariance as published.
    """
    active_groups = list(model._groups)
    shift = _public_intercept_shift(
        model,
        active_groups,
        state.profiled_factor.shape[0],
    )
    record = state.identifiability
    augmented = StructuredCovarianceAccessor(
        state.profiled_factor,
        intercept_shift=shift,
        gauge=None if record is None else record.gauge,
        weak=None if record is None else record.weak,
    )
    # The slope covariance M_ss, the slope block of the augmented inverse: no
    # raw-coordinate coefficient factor exists.
    return augmented.slopes, augmented, active_groups


def coef_covariance(model):
    """Phi-scaled Bayesian covariance for active coefficients."""
    decide_saved_record(model)
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
    decide_saved_record(model)
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


def inference_intercept_identified(inference: dict) -> bool:
    """Whether an inference record's fit has a finite intercept (false when separation moves it)."""
    record = inference.get("identifiability")
    return True if record is None else bool(record.intercept_identified())


def inference_estimated(inference: dict) -> NDArray | None:
    """Which coefficients' published values an inference record's tables show as estimates.

    Its identified ones (``inference_identified``) and those whose standard error alone its
    record withholds (``Identifiability.withheld``); an alias's and separation's show none.
    """
    identified = inference_identified(inference)
    record = inference.get("identifiability")
    if identified is None or record is None or not record.withheld:
        return identified
    estimated = np.array(identified, dtype=bool)
    withheld = [c for c in record.withheld if c >= 0]
    estimated[withheld] = record.estimate_identified()[withheld]
    return estimated


def inference_intercept_estimated(inference: dict) -> bool:
    """Whether an inference record's tables show the intercept's value (``intercept_estimated``)."""
    record = inference.get("identifiability")
    return True if record is None else bool(record.intercept_estimated())


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
    decide_saved_record(model)
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
        identifiability = _structured_identifiability(model, linear_state, W)
        if identifiability.edf is not None and identifiability.edf1 is not None:
            # the published fit's (``publish_identified_fit``)
            edf, edf1 = identifiability.edf.copy(), identifiability.edf1.copy()
        else:
            edf = linear_state.profiled_factor.inverse_operator_diagonal(
                linear_state.centered_data_operator,
            )
            edf1 = 2.0 * edf - linear_state.profiled_factor.inverse_operator_square_diagonal(
                linear_state.centered_data_operator,
            )
        edf[np.abs(edf) < 100.0 * np.finfo(float).eps] = 0.0
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
            "identifiability": identifiability,
            "coefficient_identified": identifiability.coefficient_identified(),
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
    identifiability = _dense_identifiability(
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
