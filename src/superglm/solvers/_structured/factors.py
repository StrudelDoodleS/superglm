"""Shared pieces of the structured factors' REML derivative traces, and retired names.

The factors themselves live beside their layouts: the nested chain
(``nested``), the ``fs`` block leaves (``block_leaves``) and the ``sz``
balance tree (``balance_tree``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Any

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


# The Schur-complement factor families retired by the one engine (design
# §3.12): the block-Schur pair (stage 2) and the scalar pair (stage 4), whose
# random effects every fit now factors as a nested chain (a lone level is a
# chain of one).  The names stay importable at their pickled paths as inert
# stand-ins so that models saved by v0.35.0 load; release 0.37.0 may drop them.
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
