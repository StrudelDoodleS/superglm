"""Private helpers for multi-order spline penalty components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from numpy.typing import NDArray


def build_multi_m_components(
    *,
    x: NDArray,
    m_orders: tuple[int, ...],
    build_penalty_for_order: Callable[[int], NDArray],
    apply_constraints: Callable[[Any, NDArray], tuple[Any, NDArray, int, NDArray | None]],
    apply_identifiability: Callable[
        [NDArray, NDArray, NDArray | None], tuple[NDArray, int, NDArray | None]
    ],
) -> list[tuple[str, NDArray]]:
    """Build per-order penalty components through shared constraint projections."""
    components: list[tuple[str, NDArray]] = []
    for order in m_orders:
        omega_raw = build_penalty_for_order(order)
        _, omega_c, _, constraint_proj = apply_constraints(None, omega_raw)
        omega_c, _, _ = apply_identifiability(x, omega_c, constraint_proj)
        components.append((f"d{order}", omega_c))
    return components


__all__ = ["build_multi_m_components"]
