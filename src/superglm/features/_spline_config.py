"""Configuration and validation helpers for spline feature specs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from superglm.types import LambdaPolicy


def validate_m_orders(spec: Any) -> None:
    """Validate the configured derivative penalty orders."""
    if len(spec._m_orders) > 1 and not spec._multi_m_supported:
        raise NotImplementedError(
            f"{type(spec).__name__} does not support multi-order penalties "
            f"(m tuple). Use a single m value."
        )
    if spec._max_penalty_order is not None:
        for order in spec._m_orders:
            if order > spec._max_penalty_order:
                raise ValueError(
                    f"{type(spec).__name__} supports penalty orders "
                    f"up to {spec._max_penalty_order}, got m={order}."
                )


def validate_m_orders_build(spec: Any) -> None:
    """Validate m orders after knot placement fixes the basis dimension."""
    for order in spec._m_orders:
        if order >= spec._n_basis:
            raise ValueError(
                f"Penalty order m={order} requires at least {order + 1} basis functions, "
                f"but this spline has n_basis={spec._n_basis}. "
                f"Increase n_knots or reduce m."
            )


def validate_select(spec: Any) -> None:
    """Validate select=True against the current spline capability policy."""
    if not spec.select:
        return
    if not spec._select_compatible(spec._m_orders):
        if not spec._select_supported:
            raise NotImplementedError(f"select=True is not supported for {type(spec).__name__}.")
        raise NotImplementedError(
            f"select=True with m={spec._m_orders} is not supported for "
            f"{type(spec).__name__}. "
            f"This is a current capability policy, not a mathematical impossibility."
        )


def initialize_spec(
    spec: Any,
    *,
    n_knots: int,
    degree: int,
    knot_strategy: str,
    penalty: str,
    knots: ArrayLike | None,
    discrete: bool | None,
    n_bins: int | None,
    extrapolation: str,
    boundary: tuple[float, float] | None,
    knot_alpha: float,
    select: bool,
    monotone: str | None,
    monotone_mode: str,
    m: int | tuple[int, ...],
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None,
) -> None:
    """Initialize a spline spec's public config and mutable build-time state."""
    if monotone is not None and monotone not in ("increasing", "decreasing"):
        raise ValueError(f"monotone must be None, 'increasing', or 'decreasing', got {monotone!r}")
    if monotone_mode not in ("postfit", "fit"):
        raise ValueError(f"monotone_mode must be 'postfit' or 'fit', got {monotone_mode!r}")

    spec.monotone = monotone
    spec.monotone_mode = monotone_mode
    spec.select = select

    spec._m_orders = (m,) if isinstance(m, int) else tuple(m)
    if not all(isinstance(order, int) and order >= 1 for order in spec._m_orders):
        raise ValueError(f"m must contain positive integers, got {m}")
    validate_m_orders(spec)
    validate_select(spec)

    if knots is not None:
        explicit_knots = np.asarray(knots, dtype=np.float64).ravel()
        if explicit_knots.ndim != 1 or len(explicit_knots) < 1:
            raise ValueError("knots must be a non-empty 1D array")
        if not np.all(np.diff(explicit_knots) > 0):
            raise ValueError("knots must be strictly increasing")
        spec.n_knots = len(explicit_knots)
        spec._explicit_knots = explicit_knots
    else:
        spec.n_knots = n_knots
        spec._explicit_knots = None

    spec.degree = degree
    spec.knot_strategy = knot_strategy
    spec.penalty = penalty
    spec.discrete = discrete
    spec.n_bins = n_bins
    if extrapolation not in {"clip", "extend", "error"}:
        raise ValueError(
            f"extrapolation must be one of ('clip', 'extend', 'error'), got {extrapolation!r}"
        )
    spec.extrapolation = extrapolation
    spec.knot_alpha = knot_alpha
    spec._explicit_boundary = _coerce_boundary(boundary, spec._explicit_knots)

    _initialize_runtime_state(spec, knot_strategy, lambda_policy)


def _coerce_boundary(
    boundary: tuple[float, float] | None,
    explicit_knots: NDArray | None,
) -> tuple[float, float] | None:
    """Validate and normalize an explicit fit boundary, if any."""
    if boundary is None:
        return None

    lo_bound, hi_bound = float(boundary[0]), float(boundary[1])
    if lo_bound >= hi_bound:
        raise ValueError(f"boundary must satisfy lo < hi, got boundary=({lo_bound}, {hi_bound})")
    if explicit_knots is not None:
        if explicit_knots[0] <= lo_bound or explicit_knots[-1] >= hi_bound:
            raise ValueError(
                f"explicit knots must lie strictly inside boundary=({lo_bound}, {hi_bound}), "
                f"got knots in [{explicit_knots[0]}, {explicit_knots[-1]}]"
            )
    return (lo_bound, hi_bound)


def _initialize_runtime_state(
    spec: Any,
    knot_strategy: str,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None,
) -> None:
    """Initialize mutable build-time state on a spline spec."""
    spec._knots = np.array([])
    spec._n_basis = 0
    spec._lo = 0.0
    spec._hi = 1.0
    spec._knot_strategy_actual = knot_strategy
    spec._R_inv = None
    spec._interaction_projection = None
    spec._basis_lo = None
    spec._basis_hi = None
    spec._basis_d1_lo = None
    spec._basis_d1_hi = None
    spec._U_null = None
    spec._U_range = None
    spec._omega_range = None
    spec._penalty_components = None
    spec._lambda_policy = lambda_policy
