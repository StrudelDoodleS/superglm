"""Private Hessian factor protocol and dense reference adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from superglm.types import PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers.structured import SymmetricBlockOperator


def _component_indices(component: PenaltyComponent, size: int) -> NDArray[np.intp]:
    """Return validated global coefficient indices for one penalty component."""
    start, stop, step = component.group_sl.indices(size)
    indices = np.arange(start, stop, step, dtype=np.intp)
    if step != 1:
        raise ValueError(f"Penalty component {component.name!r} must use a unit-step slice.")
    return indices


def _component_omega(component: PenaltyComponent, size: int) -> NDArray:
    """Return a dense penalty block when the component is not implicit identity."""
    indices = _component_indices(component, size)
    if component.penalty_kind == "identity":
        raise ValueError("Implicit identity penalties do not have a dense matrix.")
    if component.omega_ssp is None:
        raise ValueError(f"Dense penalty component {component.name!r} has no solver-space matrix.")
    omega = np.asarray(component.omega_ssp, dtype=np.float64)
    if omega.shape != (len(indices), len(indices)):
        raise ValueError(
            f"Penalty component {component.name!r} has shape {omega.shape}; "
            f"expected ({len(indices)}, {len(indices)})."
        )
    return omega


@runtime_checkable
class HessianFactor(Protocol):
    """Operations REML and inference require from a penalized Hessian."""

    shape: tuple[int, int]
    backend: str

    def solve(self, rhs: NDArray) -> NDArray: ...

    def logdet(self) -> float: ...

    def selected_inverse_block(self, indices: NDArray) -> NDArray: ...

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray: ...

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float: ...

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float: ...

    def trace_inverse_operator(self, operator: SymmetricBlockOperator) -> float: ...


class DenseHessianFactor:
    """Reference factor backed by the current dense inverse."""

    backend = "dense"

    def __init__(self, *, inverse: NDArray, log_det: float):
        self.inverse = np.asarray(inverse, dtype=np.float64)
        if self.inverse.ndim != 2 or self.inverse.shape[0] != self.inverse.shape[1]:
            raise ValueError("inverse must be a square matrix.")
        self.shape = self.inverse.shape
        self._log_det = float(log_det)

    def solve(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        return self.inverse @ values

    def logdet(self) -> float:
        return self._log_det

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        return self.inverse[np.ix_(selected, selected)]

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        return np.diag(self.inverse)[selected]

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        """Return ``trace(H^-1 Omega)`` using the existing dense inverse."""
        indices = _component_indices(component, self.shape[0])
        if component.penalty_kind == "identity":
            return float(np.sum(np.diag(self.inverse)[indices]))
        inverse_block = self.inverse[np.ix_(indices, indices)]
        return float(np.trace(inverse_block @ _component_omega(component, self.shape[0])))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        """Return the scaled REML Hessian trace for two penalty components."""
        left_indices = _component_indices(left, self.shape[0])
        right_indices = _component_indices(right, self.shape[0])
        right_left = self.inverse[np.ix_(right_indices, left_indices)]
        left_right = self.inverse[np.ix_(left_indices, right_indices)]
        if left.penalty_kind != "identity":
            right_left = right_left @ _component_omega(left, self.shape[0])
        if right.penalty_kind != "identity":
            left_right = left_right @ _component_omega(right, self.shape[0])
        return float(left_scale * right_scale * np.trace(right_left @ left_right))

    def trace_inverse_operator(self, operator: SymmetricBlockOperator) -> float:
        """Return ``trace(H^-1 O)`` from compact symmetric operator blocks."""
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        inverse_aa = self.inverse[np.ix_(operator.small_indices, operator.small_indices)]
        inverse_ba = self.inverse[np.ix_(operator.structured_indices, operator.small_indices)]
        inverse_bb_diagonal = np.diag(self.inverse)[operator.structured_indices]
        return float(
            np.trace(inverse_aa @ operator.A)
            + 2.0 * np.sum(inverse_ba * operator.C)
            + inverse_bb_diagonal @ operator.d
        )
