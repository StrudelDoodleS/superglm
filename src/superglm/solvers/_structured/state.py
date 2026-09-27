"""Retained support and linear-system state for structured fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from superglm._reporting_state import (
    FactorSmoothLevelSupport,
    StructuredLevelSupport,
)
from superglm.solvers._structured.factors import (
    BlockSchurFactor,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
)
from superglm.solvers._structured.moments import (
    BlockStructuredSystem,
    NestedStructuredSystem,
    ScalarStructuredSystem,
    SumToZeroBlockStructuredSystem,
)
from superglm.solvers._structured.nested import (
    NestedPenalizedOperator,
    NestedSchurFactor,
    ProfiledNestedSchurFactor,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
)

if TYPE_CHECKING:
    from superglm.solvers.sum_to_zero import (
        ProfiledSumToZeroBlockFactor,
        SumToZeroBlockFactor,
    )


@dataclass(frozen=True)
class StructuredLinearSystemState:
    """Authoritative compact factors and moments retained after a fit."""

    coefficient_factor: (
        ScalarSchurFactor | BlockSchurFactor | SumToZeroBlockFactor | NestedSchurFactor
    )
    profiled_factor: (
        ProfiledScalarSchurFactor
        | ProfiledBlockSchurFactor
        | ProfiledSumToZeroBlockFactor
        | ProfiledNestedSchurFactor
    )
    augmented_factor: (
        ScalarSchurFactor | BlockSchurFactor | SumToZeroBlockFactor | NestedSchurFactor
    )
    system: (
        ScalarStructuredSystem
        | BlockStructuredSystem
        | SumToZeroBlockStructuredSystem
        | NestedStructuredSystem
    )
    penalized_operator: (
        SymmetricBlockOperator
        | BlockSymmetricOperator
        | SumToZeroBlockOperator
        | NestedPenalizedOperator
    )
    centered_data_operator: CenteredBlockOperator
    support_totals: dict[
        str,
        StructuredLevelSupport | FactorSmoothLevelSupport,
    ]
    backend: str = "structured"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.coefficient_factor.shape != self.system.operator.shape:
            raise ValueError("Coefficient factor does not match the structured system.")
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
