"""Shared build-path helpers for spline feature specs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupInfo, TensorMarginalInfo


def build_group_info(
    spec: Any,
    x: NDArray,
    sample_weight: NDArray | None = None,
) -> GroupInfo | list[GroupInfo]:
    """Build the main GroupInfo payload for a spline spec."""
    del sample_weight
    if spec.monotone is not None and spec.monotone_mode == "fit":
        if not hasattr(spec, "_build_monotone_constraints_raw") and not hasattr(
            spec, "_build_scop_reparameterization"
        ):
            raise NotImplementedError(
                f"{type(spec).__name__} does not support "
                f"monotone_mode='fit'. Use monotone_mode='postfit'."
            )
        if spec.select:
            raise NotImplementedError(
                "Monotone fit-time constraints are not supported with "
                "select=True. Use select=False or monotone_mode='postfit'."
            )

    x = np.asarray(x, dtype=np.float64).ravel()
    spec._place_knots(x)
    spec._validate_m_orders_build()
    basis = spec._basis_matrix(x).tocsr()

    if spec.select:
        return spec._build_select(x, basis)

    omega = spec._build_penalty()
    if (
        spec.monotone is not None
        and spec.monotone_mode == "fit"
        and hasattr(spec, "_build_scop_reparameterization")
    ):
        basis_dense = basis.toarray() if hasattr(basis, "toarray") else basis
        centered_basis, scop_penalty, scop_reparam = spec._build_scop_reparameterization(
            basis_dense, omega
        )
        n_cols_scop = centered_basis.shape[1]
        return GroupInfo(
            columns=centered_basis,
            n_cols=n_cols_scop,
            penalty_matrix=scop_penalty,
            reparametrize=False,
            penalized=True,
            scop_reparameterization=scop_reparam,
            monotone_engine="scop",
        )

    basis, omega, n_cols, projection = spec._apply_constraints(basis, omega)
    omega, n_cols, projection = spec._apply_identifiability(x, omega, projection)
    spec._interaction_projection = projection

    penalty_components = None
    if len(spec._m_orders) > 1:
        penalty_components = spec._build_multi_m_components(x, basis, projection)
        omega = sum(component_omega for _, component_omega in penalty_components)

    constraints = None
    monotone_engine = None
    raw_to_solver_map = None
    if spec.monotone is not None and spec.monotone_mode == "fit":
        constraints = spec._build_monotone_constraints_raw()
        if projection is not None:
            constraints = constraints.compose(projection)
        monotone_engine = "qp"
        raw_to_solver_map = projection

    info = GroupInfo(
        columns=basis,
        n_cols=n_cols,
        penalty_matrix=omega,
        reparametrize=(spec.penalty == "ssp"),
        projection=projection,
        penalty_components=penalty_components,
        constraints=constraints,
        monotone_engine=monotone_engine,
        raw_to_solver_map=raw_to_solver_map,
    )
    if spec._lambda_policy is not None and info.penalty_components is None:
        info.penalty_components = [("wiggle", info.penalty_matrix)]
        info.component_types = {"wiggle": "difference"}
    info.lambda_policies = spec._resolve_lambda_policies(info)
    return info


def build_knots_and_penalty(
    spec: Any,
    x: NDArray,
    sample_weight: NDArray | None = None,
) -> tuple[NDArray, int, NDArray | None]:
    """Place knots and return projected penalty info without building the full basis."""
    del sample_weight
    x = np.asarray(x, dtype=np.float64).ravel()
    spec._place_knots(x)
    spec._validate_m_orders_build()
    omega = spec._build_penalty()
    _, omega_constrained, n_cols, projection = spec._apply_constraints(None, omega)

    if spec.select:
        if len(spec._m_orders) > 1:
            max_order = max(spec._m_orders)
            omega_for_eigen = spec._build_penalty_for_order(max_order)
            _, omega_for_select, _, _ = spec._apply_constraints(None, omega_for_eigen)
            spec._eigendecompose_select(omega_for_select, projection)
        else:
            spec._eigendecompose_select(omega_constrained, projection)
        spec._interaction_projection = spec._identifiability_projection(x, projection)
        return omega_constrained, n_cols, projection

    omega_ident, n_cols, projection = spec._apply_identifiability(x, omega_constrained, projection)
    spec._interaction_projection = projection

    if len(spec._m_orders) > 1:
        spec._penalty_components = spec._build_multi_m_components(x, None, projection)
        omega_ident = sum(component_omega for _, component_omega in spec._penalty_components)
    else:
        spec._penalty_components = None

    return omega_ident, n_cols, projection


def tensor_marginal_info(spec: Any, x: NDArray) -> TensorMarginalInfo:
    """Compute tensor-product marginal ingredients for an already-built spline spec."""
    if not spec._tensor_supported:
        raise NotImplementedError(
            f"{type(spec).__name__} does not support tensor marginal ingredients. "
            f"Use kind='cr' or kind='ps' for tensor product interactions."
        )

    reasons: list[str] = []
    if spec.select:
        reasons.append("select=True")
    if len(spec._m_orders) > 1:
        reasons.append(f"m={spec._m_orders}")
    if reasons:
        detail = " and ".join(reasons)
        raise NotImplementedError(
            f"Tensor interactions require single-penalty parent smooths, but "
            f"{type(spec).__name__} was configured with {detail}. "
            "This matches the mgcv te()/ti() marginal-smooth contract."
        )

    x = np.asarray(x, dtype=np.float64).ravel()
    basis_raw = spec._raw_basis_matrix(x)
    omega = spec._build_penalty()
    _, omega_constrained, _, projection_constraints = spec._apply_constraints(None, omega)

    if projection_constraints is not None:
        basis_constrained = basis_raw @ projection_constraints
    else:
        basis_constrained = basis_raw

    centered_direction = basis_constrained.sum(axis=0)
    centered_norm = np.linalg.norm(centered_direction)
    if centered_norm < 1e-12:
        projection_ident = np.eye(basis_constrained.shape[1])
    else:
        centered_direction = centered_direction / centered_norm
        q, _ = np.linalg.qr(centered_direction[:, None], mode="complete")
        projection_ident = q[:, 1:]

    basis = basis_constrained @ projection_ident
    penalty = projection_ident.T @ omega_constrained @ projection_ident
    if projection_constraints is not None:
        projection = projection_constraints @ projection_ident
    else:
        projection = projection_ident

    return TensorMarginalInfo(
        basis=basis,
        penalty=penalty,
        knots=spec._knots.copy(),
        lo=spec._lo,
        hi=spec._hi,
        projection=projection,
        K_eff=projection.shape[1],
        degree=spec.degree,
    )
