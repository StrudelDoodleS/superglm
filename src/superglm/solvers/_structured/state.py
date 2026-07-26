"""Retained support and linear-system state for structured fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.solvers._structured.factors import (
    BlockSchurFactor,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
)
from superglm.solvers._structured.moments import (
    BlockStructuredSystem,
    ScalarStructuredSystem,
    SumToZeroBlockStructuredSystem,
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
class StructuredLevelSupport:
    """Compact training support retained for one all-level structured term."""

    count: NDArray
    fit_weight: NDArray
    information: NDArray
    unpooled_effect: NDArray | None = None

    def __post_init__(self) -> None:
        expected_shape: tuple[int, ...] | None = None
        for name, dtype in (
            ("count", np.int64),
            ("fit_weight", np.float64),
            ("information", np.float64),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if expected_shape is None:
                expected_shape = values.shape
            elif values.shape != expected_shape:
                raise ValueError("Structured support arrays must have identical shapes.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.unpooled_effect is not None:
            unpooled = np.array(self.unpooled_effect, dtype=np.float64, copy=True)
            if unpooled.shape != expected_shape:
                raise ValueError("unpooled_effect must match the structured support shape.")
            unpooled.setflags(write=False)
            object.__setattr__(self, "unpooled_effect", unpooled)


@dataclass(frozen=True)
class FactorSmoothLevelSupport:
    """Compact row support and local Fisher information for a factor smooth."""

    count: NDArray
    fit_weight: NDArray
    information: NDArray

    def __post_init__(self) -> None:
        count = np.array(self.count, dtype=np.int64, copy=True)
        fit_weight = np.array(self.fit_weight, dtype=np.float64, copy=True)
        information = np.array(self.information, dtype=np.float64, copy=True)
        if count.ndim != 1 or fit_weight.shape != count.shape:
            raise ValueError("FactorSmooth count and fit_weight must be aligned vectors.")
        if (
            information.ndim != 3
            or information.shape[0] != len(count)
            or information.shape[1] != information.shape[2]
        ):
            raise ValueError(
                "FactorSmooth information must have shape (n_levels, block_size, block_size)."
            )
        if not np.allclose(
            information,
            information.transpose(0, 2, 1),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("FactorSmooth local information blocks must be symmetric.")
        for values in (count, fit_weight, information):
            values.setflags(write=False)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "fit_weight", fit_weight)
        object.__setattr__(self, "information", information)


@dataclass(frozen=True)
class StructuredLinearSystemState:
    """Authoritative compact factors and moments retained after a fit."""

    coefficient_factor: ScalarSchurFactor | BlockSchurFactor | SumToZeroBlockFactor
    profiled_factor: (
        ProfiledScalarSchurFactor | ProfiledBlockSchurFactor | ProfiledSumToZeroBlockFactor
    )
    augmented_factor: ScalarSchurFactor | BlockSchurFactor | SumToZeroBlockFactor
    system: ScalarStructuredSystem | BlockStructuredSystem | SumToZeroBlockStructuredSystem
    penalized_operator: SymmetricBlockOperator | BlockSymmetricOperator | SumToZeroBlockOperator
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
