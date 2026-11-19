"""Retained support and linear-system state for structured fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    There is no raw-coordinate coefficient factor: the slope covariance is
    ``M_ss``, the slope block of the augmented inverse (``profiled_factor``).
    Older pickles' ``coefficient_factor`` or ``fallback_reason`` entries are
    restored as unread attributes.
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


def centred_data_operator(system) -> CenteredBlockOperator:
    """``system``'s data operator centred on its working-weighted mean, ``X~' W X~``.

    An ``fs`` or ``sz`` leaf system serves it on its ``c0``-shifted moments
    (``centred_data_operator``), which round at the
    columns' spread; every other system centres its raw moments on ``xtw /
    sum_w``.
    """
    if isinstance(system, FactorSmoothLeafSystem | SumToZeroLeafSystem):
        return system.centred_data_operator
    xtw = np.empty(system.operator.shape[0], dtype=np.float64)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    return CenteredBlockOperator(
        raw=system.operator, cross=xtw, total=system.sum_w, center=xtw / system.sum_w
    )
