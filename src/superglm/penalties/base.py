"""Penalty and Flavor protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from superglm.types import GroupSlice


@runtime_checkable
class Penalty(Protocol):
    """Protocol for penalty functions.

    The solver calls prox() in the inner loop and eval() for diagnostics.
    Penalties are passed as objects so different penalties can be swapped
    without changing the solver.
    """

    @property
    def lambda1(self) -> float: ...

    @property
    def flavor(self) -> Flavor | None: ...

    def prox(self, beta: NDArray, groups: list[GroupSlice], step: float) -> NDArray:
        """Proximal operator: argmin_z  step * P(z) + 0.5 * ||beta - z||^2."""
        ...

    def prox_group(self, bg: NDArray, group: GroupSlice, step: float) -> NDArray:
        """Proximal operator for a single group's coefficients.

        Parameters
        ----------
        bg : NDArray of shape (p_g,)
            Candidate coefficients for this group (already gradient-stepped).
        group : GroupSlice
            Group metadata (weight, size, etc.).
        step : float
            Step size (1 / L_g for this group).

        Returns
        -------
        NDArray of shape (p_g,)
        """
        ...

    def eval(self, beta: NDArray, groups: list[GroupSlice]) -> float:
        """Evaluate penalty value P(beta)."""
        ...


@runtime_checkable
class Flavor(Protocol):
    """Protocol for penalty modifiers (e.g. adaptive weighting).

    Flavors adjust group weights based on an initial estimate before
    the solver runs its main fit.
    """

    def adjust_weights(
        self,
        groups: list[GroupSlice],
        beta_init: NDArray,
    ) -> list[GroupSlice]:
        """Return new GroupSlice list with modified weights."""
        ...
