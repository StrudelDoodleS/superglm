"""Private select/lambda-policy helpers for spline feature specs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.features._spline_identifiability import (
    build_identifiability_projection_for_spec,
    null_explicit_coordinates,
    term_null_basis,
)
from superglm.features._spline_ranges import _REML_RANK_THRESHOLD
from superglm.solvers.rank import SHARED_RANK_POLICY, _eigensolver_relative_bar
from superglm.types import GroupInfo, LambdaPolicy


def _spread_refusal(subject: str, ratio: float) -> ValueError:
    return ValueError(
        f"{subject} cannot split this penalty: its knot intervals differ so much in "
        f"width that the curvature it puts on the widest is {ratio:.1e} "
        "of the curvature on the narrowest, which double precision cannot hold beside "
        "it. The widest interval is usually the tail of a heavy-tailed column under "
        'quantile knots. Pass kind="ps", whose penalty double precision can hold on '
        "any knots, transform the column (for example, take its logarithm), or place "
        "the knots yourself."
    )


def root_select_split(
    omega_c: NDArray,
    Z: NDArray | None,
    *,
    null_raw: NDArray,
    root_raw: NDArray,
    n_basis: int,
    structural: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Split a select=True penalty from its exact null basis and pre-congruence root.

    The right singular vectors of the root on the orthogonal complement of
    the polynomials diagonalise its range without squaring its condition.
    A different constrained nullity, or a zero penalty (a column with one
    value), is handled by the existing split.
    """
    if not np.any(root_raw):
        return None
    if structural is not None:
        structural_null = _null_mask(np.linalg.eigvalsh(structural))
        if int(np.sum(structural_null)) != 2:
            return None
    U_null_raw = term_null_basis(null_raw, Z, 2)
    if U_null_raw is None:
        return None

    ones_c = (Z.T @ np.ones(n_basis)) if Z is not None else np.ones(omega_c.shape[0])
    ones_in_null = U_null_raw.T @ ones_c
    ones_norm = float(np.linalg.norm(ones_in_null))
    if ones_norm <= 0.0:
        return None
    a, b = ones_in_null / ones_norm
    U_null_1d = U_null_raw @ np.array([[-b], [a]], dtype=np.float64)

    Q_range = null_explicit_coordinates(U_null_raw)[:, 2:]
    n_range = Q_range.shape[1]
    if n_range == 0:
        return None
    root_c = root_raw if Z is None else root_raw @ Z
    R_range = root_c @ Q_range
    _, singular, vh = np.linalg.svd(R_range, full_matrices=False)
    root_scale = max(
        float(singular[0]) if singular.size else 0.0,
        float(np.linalg.norm(root_raw, 2)),
    )
    min_resolved = (
        SHARED_RANK_POLICY.certification_band
        * _eigensolver_relative_bar(max(root_raw.shape))
        * root_scale
    )
    if (
        singular.size != n_range
        or singular[0] <= 0.0
        or singular[-1] <= min_resolved
        or not np.all(singular**2 > 0.0)
    ):
        ratio = (
            float((singular[-1] / singular[0]) ** 2)
            if singular.size == n_range and singular[0] > 0.0
            else 0.0
        )
        raise _spread_refusal("select=True", ratio)

    U_range_c = Q_range @ vh.T
    U_null = Z @ U_null_1d if Z is not None else U_null_1d
    U_range = Z @ U_range_c if Z is not None else U_range_c
    return U_null, U_range, np.diag(singular**2)


def eigendecompose_select(
    omega_c: NDArray,
    Z: NDArray | None,
    *,
    n_basis: int,
    spline_kind: str,
    structural: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Eigendecompose the constrained penalty for select=True splitting.

    An eigenvalue is null when it is at most ``_REML_RANK_THRESHOLD`` of the largest,
    the rule REML ranks the same penalty by.  The cut is relative because a penalty
    scales with a power of the feature's units, and it sits far above the
    eigensolver's resolution of ``n * eps`` of the largest eigenvalue (*LAPACK
    Users' Guide*, 3rd ed., section 4.7).

    A relative cut fixes units but not the spread within a penalty: an integrated
    derivative penalty (``cr``, ``bs``) gives a direction confined to a knot interval
    of width ``w`` an eigenvalue of order ``w**-3``, so quantile knots on a
    heavy-tailed column can put a tail direction under the cut.  For those kinds
    ``structural``, the constrained penalty with each interval's block at unit norm,
    decides the split: it has the penalty's null space (polynomials of degree below
    ``m``) without the spread (Wood, Pya and Saefken, JASA 111, 2016, section
    3.1.1).  Where the real penalty's spectrum shows the same nullity, its own
    eigenvectors are kept; otherwise the range is the real penalty restricted to the
    structural range and diagonalised there, which ``_certified_range`` refuses when
    the penalty no longer resolves a range direction above round-off.
    """
    eigvals, eigvecs = np.linalg.eigh(omega_c)
    null_mask = _null_mask(eigvals)
    if structural is not None:
        structural_values, structural_vectors = np.linalg.eigh(structural)
        structural_null = _null_mask(structural_values)
        _require_two_null(structural_null, spline_kind)
    if structural is None or np.sum(null_mask) == np.sum(structural_null):
        _require_two_null(null_mask, spline_kind)
        U_null_raw = eigvecs[:, null_mask]
        U_range = eigvecs[:, ~null_mask]
        omega_range = np.diag(eigvals[~null_mask])
    else:
        U_null_raw = structural_vectors[:, structural_null]
        U_range, omega_range = _certified_range(omega_c, structural_vectors[:, ~structural_null])

    ones_c = (Z.T @ np.ones(n_basis)) if Z is not None else np.ones(omega_c.shape[0])
    ones_in_null = U_null_raw.T @ ones_c
    ones_in_null /= np.linalg.norm(ones_in_null)
    U_null_centered = U_null_raw - U_null_raw @ np.outer(ones_in_null, ones_in_null)
    u, _, _ = np.linalg.svd(U_null_centered, full_matrices=False)
    U_null_1d = u[:, :1]

    U_null = Z @ U_null_1d if Z is not None else U_null_1d
    U_range = Z @ U_range if Z is not None else U_range
    return U_null, U_range, omega_range


def _null_mask(eigvals: NDArray) -> NDArray:
    return eigvals <= _REML_RANK_THRESHOLD * max(eigvals[-1], 0.0)


def _require_two_null(null_mask: NDArray, spline_kind: str) -> None:
    n_null = int(np.sum(null_mask))
    if n_null != 2:
        raise ValueError(
            f"select=True requires exactly 2 null eigenvalues in the "
            f"constrained penalty, got {n_null}. "
            f"Spline kind {spline_kind} may not support select=True."
        )


def _certified_range(
    omega_c: NDArray, basis: NDArray, *, subject: str = "select=True"
) -> tuple[NDArray, NDArray]:
    """The penalty on the structural range, diagonalised, where binary64 certifies it definite.

    In exact arithmetic ``M = basis' omega_c basis`` is positive definite, but an
    integrated derivative penalty on skewed knots puts its tail direction at a tiny
    fraction of the bulk.  The split is refused only where the computed ``M`` cannot
    certify that direction positive: Rump's test
    (``reml.multi_penalty._certifies_positive_definite``) holds every symmetric
    matrix within the formation enclosure of ``M`` positive definite, so a refusal
    means binary64 cannot certify the direction penalised at all.  The computed
    eigenvalues must also be positive for the diagonal returned.
    """
    from superglm.reml.multi_penalty import (
        _certified_congruence,
        _certifies_positive_definite,
    )

    product, radius = _certified_congruence(basis, omega_c)
    values, vectors = np.linalg.eigh(product)
    if not (values[0] > 0 and _certifies_positive_definite(product, radius)):
        ratio = max(values[0], 0.0) / values[-1]
        raise _spread_refusal(subject, ratio)
    return basis @ vectors, np.diag(values)


def resolve_lambda_policies(
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None,
    info: GroupInfo,
    m_orders: tuple[int, ...],
    select: bool,
) -> dict[str, LambdaPolicy] | None:
    """Resolve lambda_policy into a policy for each penalty ``info`` carries.

    A policy may name the penalties the build carries, ``"wiggle"`` for a
    single unnamed one. A column whose penalty is identically zero (one value
    under ``kind="bs"``) carries none, and accepts the names the declaration
    gives any other column: ``"wiggle"`` for one derivative order, else
    ``"d{order}"`` for each, with ``"null"`` beside them under ``select=True``.
    """
    if lambda_policy is None:
        return None

    if info.penalty_components is not None:
        built_names = {name for name, _ in info.penalty_components}
    else:
        built_names = {"wiggle"}

    if isinstance(lambda_policy, LambdaPolicy):
        return {name: lambda_policy for name in built_names}

    if info.penalty_components is not None or info.penalty_matrix is not None:
        valid_names = built_names
    else:
        valid_names = {"wiggle"} if len(m_orders) == 1 else {f"d{order}" for order in m_orders}
        if select:
            valid_names.add("null")
    policy_dict = lambda_policy
    unknown = set(policy_dict) - valid_names
    if unknown:
        raise ValueError(
            f"lambda_policy contains unknown component names: {unknown}. "
            f"Valid names: {sorted(valid_names)}"
        )
    return {name: policy_dict.get(name, LambdaPolicy.estimate()) for name in built_names}


def build_select_group_info(spec: Any, B: Any, Z: NDArray | None) -> GroupInfo:
    """Assemble selection and wiggliness penalties in the split spline coordinates."""
    projection = np.column_stack((spec._U_null, spec._U_range))
    width = projection.shape[1]
    selection = np.zeros((width, width))
    selection[0, 0] = 1.0
    components = [("null", selection)]
    orders = spec._m_orders
    exact_split = spec._select_from_root
    for order in orders:
        if order == max(orders):
            penalty = np.zeros_like(selection)
            penalty[1:, 1:] = spec._omega_range
        elif exact_split:
            # Building the order also resolves the actual difference operator
            # on uneven knots and checks polynomial-range constraints.
            spec._build_penalty_for_order(order)
            root = spec._penalty_root_for_order(order) @ projection
            null = spec._penalty_null_basis(order)
            q = np.linalg.qr(null)[0]
            line = projection[:, 0]
            if np.linalg.norm(line - q @ (q.T @ line)) <= SHARED_RANK_POLICY.factor_rcond:
                root[:, 0] = 0.0
            penalty = root.T @ root
        else:
            raw = spec._build_penalty_for_order(order)
            _, constrained, _, _ = spec._apply_constraints(None, raw)
            local = projection if Z is None else np.linalg.lstsq(Z, projection, rcond=None)[0]
            penalty = local.T @ constrained @ local
        components.append(("wiggle" if len(orders) == 1 else f"d{order}", penalty))
    info = GroupInfo(
        columns=B,
        n_cols=width,
        penalty_matrix=sum(omega for _, omega in components),
        reparametrize=True,
        projection=projection,
        penalty_components=components,
        component_types={"null": "selection"},
        penalty_in_term_coordinates=exact_split,
    )
    info.lambda_policies = spec._resolve_lambda_policies(info)
    return info


def build_select(
    spec: Any,
    x: NDArray,
    B: Any,
    geometry_weight: NDArray | None = None,
) -> GroupInfo:
    """Build select=True GroupInfo for a spline spec."""
    if len(spec._m_orders) == 1:
        omega_for_eigen = spec._build_penalty()
    else:
        max_order = max(spec._m_orders)
        omega_for_eigen = spec._build_penalty_for_order(max_order)
    _, omega_c, _, Z = spec._apply_constraints(None, omega_for_eigen)

    spec._interaction_projection = build_identifiability_projection_for_spec(
        spec,
        x,
        Z,
        geometry_weight,
    )
    spec._eigendecompose_select(omega_c, Z)
    return build_select_group_info(spec, B, Z)


__all__ = [
    "build_select",
    "build_select_group_info",
    "eigendecompose_select",
    "resolve_lambda_policies",
    "root_select_split",
]
