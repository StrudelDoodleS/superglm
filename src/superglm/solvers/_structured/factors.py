"""Shared pieces of the structured factors: the profiled adapters, REML derivative traces, retired names.

The factors themselves live beside their layouts: the nested chain
(``nested``), the ``fs`` block leaves (``block_leaves``) and the ``sz``
balance tree (``balance_tree``).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from superglm.solvers._structured.operators import (
    CompactSymmetricOperator,
    SumBlockOperator,
    _bdlr_cross_traces,
    _operator_bdlr,
)
from superglm.solvers._structured.retired import module_getattr
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers._structured.balance_tree import SumToZeroLeafSystem
    from superglm.solvers._structured.block_leaves import FactorSmoothLeafSystem

# One REML Hessian direction: the penalty component, its lambda and the
# optional W-derivative operator, together ``lambda * Omega + dH``.
DerivativeDirection = tuple[PenaltyComponent, float, CompactSymmetricOperator | None]


def _penalty_product(component: PenaltyComponent, scale: float, basis: NDArray) -> NDArray:
    """Return ``scale * Omega @ basis`` touching only the penalty's own rows."""
    size = basis.shape[0]
    indices = _component_indices(component, size)
    rows = basis[indices]
    product = np.zeros_like(basis)
    if component.penalty_kind == "identity":
        product[indices] = scale * rows
    elif component.penalty_kind == "repeated":
        omega = np.asarray(component.omega_ssp, dtype=np.float64)
        blocks = rows.reshape(int(component.repeat_count), omega.shape[0], rows.shape[1])
        product[indices] = scale * (omega @ blocks).reshape(rows.shape)
    else:
        product[indices] = scale * (_component_omega(component, size) @ rows)
    return product


def _derivative_directions(
    directions: Sequence[DerivativeDirection],
    penalty_operator: Callable[[PenaltyComponent, float], CompactSymmetricOperator],
    basis: NDArray,
    local_form: Callable[[CompactSymmetricOperator], Any],
) -> Iterator[tuple[NDArray, Any]]:
    """Yield ``(O U, O's local part)`` once per direction ``O = scale * Omega + dH``."""
    for component, scale, operator in directions:
        product = _penalty_product(component, scale, basis)
        combined = penalty_operator(component, scale)
        if operator is not None:
            product += operator.matvec(basis)
            combined = SumBlockOperator((combined, operator))
        yield product, local_form(combined)


def _block_derivative_cross_traces(
    factor: Any,
    directions: Sequence[DerivativeDirection],
) -> NDArray:
    """Pairwise derivative traces of a block-leaf factor through its ``_inverse_bdlr``."""
    inverse = factor._inverse_bdlr()
    return _bdlr_cross_traces(
        inverse,
        _derivative_directions(
            directions,
            factor._penalty_operator,
            inverse.basis,
            partial(_operator_bdlr, structured_indices=inverse.structured_indices, local_only=True),
        ),
    )


class _ProfiledInterceptFactor:
    """The slope inverse of an intercept-augmented factor, read through that factor.

    The intercept is the augmented factor's coordinate 0, so a slope index,
    penalty component or right-hand side is the augmented one shifted by one,
    and ``logdet()`` is the augmented ``logdet - log(sum_w)``.  An adapter
    sets ``augmented_factor``, ``sum_w`` and ``shape``; ``_solve_options``
    are its augmented ``solve``'s keywords.
    """

    backend = "structured"
    _solve_options: ClassVar[dict] = {"centred": True}

    @staticmethod
    def _shift_component(component: PenaltyComponent) -> PenaltyComponent:
        start, stop = component.group_sl.start, component.group_sl.stop
        if start is None or stop is None:
            raise ValueError("Penalty component slices must have explicit bounds.")
        return replace(component, group_sl=slice(start + 1, stop + 1, component.group_sl.step))

    def solve(self, rhs: NDArray) -> NDArray:
        """Return ``H_c^-1 rhs``: the augmented solve of ``[0; rhs]``, intercept row dropped."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {np.asarray(rhs).shape}."
            )
        augmented_rhs = np.zeros((self.shape[0] + 1, values.shape[1]))
        augmented_rhs[1:] = values
        solution = self.augmented_factor.solve(augmented_rhs, **self._solve_options)[1:]
        return solution[:, 0] if vector_rhs else solution

    def logdet(self) -> float:
        return float(self.augmented_factor.logdet() - np.log(self.sum_w))

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_block(np.asarray(indices, dtype=np.intp) + 1)

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        return self.augmented_factor.selected_inverse_diagonal(
            np.asarray(indices, dtype=np.intp) + 1
        )

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        return self.augmented_factor.trace_inverse_penalty(self._shift_component(component))

    def penalty_cross_trace(
        self, left: PenaltyComponent, right: PenaltyComponent, left_scale: float, right_scale: float
    ) -> float:
        return self.augmented_factor.penalty_cross_trace(
            self._shift_component(left), self._shift_component(right), left_scale, right_scale
        )


class _ProfiledLeafFactor(_ProfiledInterceptFactor):
    """The ``fs`` and ``sz`` adapters' attributes off the augmented factor, and their publication."""

    def _adopt(self, augmented_factor, sum_w: float, xtw: NDArray) -> None:
        self.augmented_factor = augmented_factor
        self.sum_w = float(sum_w)
        self.xtw = np.asarray(xtw, dtype=np.float64)
        if not np.isfinite(self.sum_w) or self.sum_w <= 0.0:
            raise ValueError("sum_w must be positive and finite.")
        if augmented_factor.shape != (len(self.xtw) + 1, len(self.xtw) + 1):
            raise ValueError("Augmented factor width does not match xtw.")
        self.shape = (len(self.xtw), len(self.xtw))
        self.mean_x = self.xtw / self.sum_w
        self.rank = max(int(augmented_factor.rank) - 1, 0)
        self.rank_truncated = self.rank < self.shape[0]
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.n_levels = augmented_factor.n_levels
        self.block_size = augmented_factor.block_size
        self.max_structured_inverse_block = augmented_factor.max_structured_inverse_block
        self.border_certificate = augmented_factor.border_certificate

    def published(self, system: FactorSmoothLeafSystem | SumToZeroLeafSystem | None = None):
        """This factor with its augmented factor on the published ``system`` (its own, published, if ``None``)."""
        augmented = self.augmented_factor
        clone = copy.copy(self)
        clone.augmented_factor = augmented.published(
            augmented.system.published() if system is None else system
        )
        return clone


# The Schur-complement factor families retired by the one engine (design
# §3.12): the block-Schur pair (stage 2) and the scalar pair (stage 4), whose
# random effects every fit now factors as a nested chain (a lone level is a
# chain of one).  The names stay importable at their pickled paths as inert
# stand-ins so that models saved by v0.35.0 load; a later minor release may
# drop them.
__getattr__ = module_getattr(
    __name__,
    frozenset(
        {
            "BlockSchurFactor",
            "ProfiledBlockSchurFactor",
            "ProfiledScalarSchurFactor",
            "ScalarSchurFactor",
        }
    ),
)
