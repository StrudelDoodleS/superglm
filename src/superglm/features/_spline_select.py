"""Private select/lambda-policy helpers for spline feature specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.features._spline_identifiability import (
    build_identifiability_projection_for_spec,
)
from superglm.features._spline_ranges import _REML_RANK_THRESHOLD
from superglm.solvers.rank import SHARED_RANK_POLICY, _eigensolver_relative_bar
from superglm.types import GroupInfo, LambdaPolicy


def eigendecompose_select(
    omega_c: NDArray,
    Z: NDArray | None,
    *,
    n_basis: int,
    spline_kind: str,
    structural: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Eigendecompose the constrained penalty for select=True splitting.

    An eigenvalue is null when it is at most ``_REML_RANK_THRESHOLD`` of the
    largest, the rule REML ranks the same penalty by, so the split agrees with
    it. The cut is relative because a penalty scales with a power of the
    feature's units (a cubic regression spline's with the inverse cube of its
    range), and it sits far above the eigensolver's resolution of ``n * eps``
    of the largest eigenvalue (*LAPACK Users' Guide*, 3rd ed., section 4.7).

    A relative cut fixes units but not the spread within a penalty: an
    integrated derivative penalty (``cr``, ``bs``) gives a direction confined
    to a knot interval of width ``w`` an eigenvalue of order ``w**-3``, so
    quantile knots on a heavy-tailed column put the tail interval's direction
    under the cut. For those kinds ``structural``, the constrained penalty with
    each interval's block at unit norm, decides the split: a positive
    reweighting of positive semidefinite blocks has their null space (the
    polynomials of degree below ``m``) without the spread. That is Wood, Pya
    and Saefken's balanced penalty (JASA 111, 2016, section 3.1.1), whose
    penalised and unpenalised spaces are those of the penalty it balances. Where the real penalty's spectrum shows the same
    nullity at the cut, its own eigenvectors are kept, so those fits are
    unchanged; otherwise the null space is the structural one and the range is
    the real penalty restricted to the structural range and diagonalised there,
    which ``_certified_range`` refuses when the penalty no longer resolves a
    range direction above round-off.
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
    """The penalty on the structural range, diagonalised, once round-off cannot hide a direction.

    In exact arithmetic ``basis' omega_c basis`` is positive definite, since
    ``basis`` spans the complement of the penalty's null space. Its computed
    eigenvalues are within ``p(n) eps ||omega_c||_2`` of the exact ones
    (*LAPACK Users' Guide*, 3rd ed., section 4.7; ``_eigensolver_relative_bar``),
    a bar that also covers forming the product. The smallest must clear it by
    the shared policy's ``certification_band``, else the tail direction's
    curvature is below what binary64 holds beside the bulk's and no rank
    decision can recover it.
    """
    values, vectors = np.linalg.eigh(basis.T @ omega_c @ basis)
    resolution = _eigensolver_relative_bar(omega_c.shape[0]) * max(values[-1], 0.0)
    if not values[0] > SHARED_RANK_POLICY.certification_band * resolution:
        ratio = max(values[0], 0.0) / values[-1]
        raise ValueError(
            f"{subject} cannot split this penalty: its knot intervals differ so much in "
            f"width that the curvature it puts on the widest is {ratio:.1e} "
            "of the curvature on the narrowest, which double precision cannot hold beside "
            "it. The widest interval is usually the tail of a heavy-tailed column under "
            'quantile knots. Pass kind="ps", whose penalty double precision can hold on '
            "any knots, transform the column (for example, take its logarithm), or place "
            "the knots yourself."
        )
    return basis @ vectors, np.diag(values)


def resolve_lambda_policies(
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None,
    info: GroupInfo,
) -> dict[str, LambdaPolicy] | None:
    """Resolve lambda_policy parameter into a per-component dict."""
    if lambda_policy is None:
        return None

    if info.penalty_components is not None:
        valid_names = {name for name, _ in info.penalty_components}
    else:
        valid_names = {"wiggle"}

    if isinstance(lambda_policy, LambdaPolicy):
        return {name: lambda_policy for name in valid_names}

    policy_dict = lambda_policy
    unknown = set(policy_dict) - valid_names
    if unknown:
        raise ValueError(
            f"lambda_policy contains unknown component names: {unknown}. "
            f"Valid names: {sorted(valid_names)}"
        )
    return {name: policy_dict.get(name, LambdaPolicy.estimate()) for name in valid_names}


def build_select_group_info(
    *,
    B: Any,
    m_orders: tuple[int, ...],
    U_null: NDArray,
    U_range: NDArray,
    omega_range: NDArray,
    Z: NDArray | None,
    build_penalty_for_order: Callable[[int], NDArray],
    apply_constraints: Callable[[Any, NDArray], tuple[Any, NDArray, int, NDArray | None]],
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None,
) -> GroupInfo:
    """Build select=True GroupInfo from the eigendecomposed constrained penalty."""
    n_null = 1
    n_range = U_range.shape[1]
    n_combined = n_null + n_range

    U_combined = np.hstack([U_null, U_range])
    U_null_c = U_null if Z is None else np.linalg.lstsq(Z, U_null, rcond=None)[0]
    U_range_c = U_range if Z is None else np.linalg.lstsq(Z, U_range, rcond=None)[0]
    U_combined_c = np.hstack([U_null_c, U_range_c])

    omega_null = np.zeros((n_combined, n_combined))
    omega_null[:n_null, :n_null] = np.eye(n_null)

    components: list[tuple[str, NDArray]] = [("null", omega_null)]
    component_types: dict[str, str] = {"null": "selection"}

    if len(m_orders) == 1:
        omega_wiggle = np.zeros((n_combined, n_combined))
        omega_wiggle[n_null:, n_null:] = omega_range
        components.append(("wiggle", omega_wiggle))
    else:
        for order in m_orders:
            omega_raw_j = build_penalty_for_order(order)
            _, omega_c_j, _, _ = apply_constraints(None, omega_raw_j)
            omega_combined_j = U_combined_c.T @ omega_c_j @ U_combined_c
            components.append((f"d{order}", omega_combined_j))

    penalty_matrix = sum(omega for _, omega in components)
    info = GroupInfo(
        columns=B,
        n_cols=n_combined,
        penalty_matrix=penalty_matrix,
        reparametrize=True,
        penalized=True,
        projection=U_combined,
        penalty_components=components,
        component_types=component_types,
    )
    info.lambda_policies = resolve_lambda_policies(lambda_policy, info)
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
    assert spec._U_null is not None
    assert spec._U_range is not None
    assert spec._omega_range is not None
    return build_select_group_info(
        B=B,
        m_orders=spec._m_orders,
        U_null=spec._U_null,
        U_range=spec._U_range,
        omega_range=spec._omega_range,
        Z=Z,
        build_penalty_for_order=(
            spec._build_penalty_for_order
            if hasattr(spec, "_build_penalty_for_order")
            else lambda order: spec._build_penalty()
        ),
        apply_constraints=spec._apply_constraints,
        lambda_policy=spec._lambda_policy,
    )


__all__ = [
    "build_select",
    "build_select_group_info",
    "eigendecompose_select",
    "resolve_lambda_policies",
]
