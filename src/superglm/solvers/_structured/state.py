"""Retained support and linear-system state for structured fits."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._reporting_state import (
    FactorSmoothLevelSupport,
    StructuredLevelSupport,
)
from superglm.solvers._structured.balance_tree import (
    ProfiledSumToZeroTreeFactor,
    SumToZeroLeafSystem,
    SumToZeroTreeFactor,
)
from superglm.solvers._structured.block_leaves import (
    FactorSmoothLeafFactor,
    FactorSmoothLeafSystem,
    ProfiledFactorSmoothLeafFactor,
)
from superglm.solvers._structured.moments import NestedStructuredSystem
from superglm.solvers._structured.nested import (
    NestedPenalizedOperator,
    NestedSchurFactor,
    ProfiledNestedSchurFactor,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SumToZeroBlockOperator,
)


@dataclass(frozen=True)
class StructuredLinearSystemState:
    """Authoritative compact factors and moments retained after a fit.

    There is no raw-coordinate coefficient factor (one-engine design §3.6,
    §3.10): the slope covariance is ``M_ss``, the slope block of the augmented
    inverse, which ``profiled_factor`` serves.  A state pickled before this
    carries a ``coefficient_factor`` entry, which unpickling restores as an
    unread attribute, and one pickled by v0.35.0 its retired factor's
    ``fallback_reason``.
    """

    profiled_factor: (
        ProfiledFactorSmoothLeafFactor | ProfiledSumToZeroTreeFactor | ProfiledNestedSchurFactor
    )
    augmented_factor: FactorSmoothLeafFactor | SumToZeroTreeFactor | NestedSchurFactor
    system: FactorSmoothLeafSystem | SumToZeroLeafSystem | NestedStructuredSystem
    penalized_operator: BlockSymmetricOperator | SumToZeroBlockOperator | NestedPenalizedOperator
    centered_data_operator: CenteredBlockOperator
    support_totals: dict[
        str,
        StructuredLevelSupport | FactorSmoothLevelSupport,
    ]
    backend: str = "structured"

    def __post_init__(self) -> None:
        if self.profiled_factor.shape != self.system.operator.shape:
            raise ValueError("Profiled factor does not match the structured system.")
        expected_augmented = self.system.operator.shape[0] + 1
        if self.augmented_factor.shape != (expected_augmented, expected_augmented):
            raise ValueError("Augmented factor does not match the structured system.")
        if self.penalized_operator.shape != self.system.operator.shape:
            raise ValueError("Penalized operator does not match the structured system.")
        if self.centered_data_operator.shape != self.system.operator.shape:
            raise ValueError("Centered data operator does not match the structured system.")
        object.__setattr__(self, "support_totals", dict(self.support_totals))


def centred_data_operator(
    system, *, row_column_norm: NDArray | None = None
) -> CenteredBlockOperator:
    """``system``'s data operator centred on its working-weighted mean, ``X~' W X~``.

    An ``fs`` or ``sz`` leaf system serves it on its ``c0``-shifted moments
    (``centred_data_operator``, one-engine design §3.2), which round at the
    columns' spread; every other system centres its raw moments on ``xtw /
    sum_w``.  ``row_column_norm`` attaches the rows' norms of the columns whose
    moments cancel (``geometry.cancelled_column_row_norms``).
    """
    if isinstance(system, FactorSmoothLeafSystem | SumToZeroLeafSystem):
        operator = system.centred_data_operator
    else:
        xtw = np.empty(system.operator.shape[0], dtype=np.float64)
        xtw[system.operator.small_indices] = system.xtw_small
        xtw[system.operator.structured_indices] = system.xtw_structured
        operator = CenteredBlockOperator(
            raw=system.operator, cross=xtw, total=system.sum_w, center=xtw / system.sum_w
        )
    if row_column_norm is not None:
        operator = dataclasses.replace(operator, row_column_norm=row_column_norm)
    return operator
