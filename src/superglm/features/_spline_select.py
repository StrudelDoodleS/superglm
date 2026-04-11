"""Private select/lambda-policy helpers for spline feature specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupInfo, LambdaPolicy


def eigendecompose_select(
    omega_c: NDArray,
    Z: NDArray | None,
    *,
    n_basis: int,
    spline_kind: str,
) -> tuple[NDArray, NDArray, NDArray]:
    """Eigendecompose the constrained penalty for select=True splitting."""
    eigvals, eigvecs = np.linalg.eigh(omega_c)
    null_mask = eigvals < 1e-10
    n_null = int(np.sum(null_mask))
    if n_null != 2:
        raise ValueError(
            f"select=True requires exactly 2 null eigenvalues in the "
            f"constrained penalty, got {n_null}. "
            f"Spline kind {spline_kind} may not support select=True."
        )

    U_null_raw = eigvecs[:, null_mask]
    U_range = eigvecs[:, ~null_mask]
    omega_range = np.diag(eigvals[~null_mask])

    ones_c = (Z.T @ np.ones(n_basis)) if Z is not None else np.ones(omega_c.shape[0])
    ones_in_null = U_null_raw.T @ ones_c
    ones_in_null /= np.linalg.norm(ones_in_null)
    U_null_centered = U_null_raw - U_null_raw @ np.outer(ones_in_null, ones_in_null)
    u, _, _ = np.linalg.svd(U_null_centered, full_matrices=False)
    U_null_1d = u[:, :1]

    U_null = Z @ U_null_1d if Z is not None else U_null_1d
    U_range = Z @ U_range if Z is not None else U_range
    return U_null, U_range, omega_range


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


def build_select(spec: Any, x: NDArray, B: Any) -> GroupInfo:
    """Build select=True GroupInfo for a spline spec."""
    if len(spec._m_orders) == 1:
        omega_for_eigen = spec._build_penalty()
    else:
        max_order = max(spec._m_orders)
        omega_for_eigen = spec._build_penalty_for_order(max_order)
    _, omega_c, _, Z = spec._apply_constraints(None, omega_for_eigen)

    spec._interaction_projection = spec._identifiability_projection(x, Z)
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
