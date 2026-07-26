"""Backend-neutral compact support retained for credibility reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, clip_mu
from superglm.group_matrix import FactorSmoothGroupMatrix, RandomEffectGroupMatrix
from superglm.links import stabilize_eta

if TYPE_CHECKING:
    from superglm.distributions import Distribution
    from superglm.group_matrix import DesignMatrix
    from superglm.links import Link
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


@dataclass(frozen=True)
class StructuredLevelSupport:
    """Compact training support retained for one all-level random effect."""

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
class ReportingSupportState:
    """Backend-independent sufficient statistics for structured term reports."""

    support_totals: dict[
        str,
        StructuredLevelSupport | FactorSmoothLevelSupport,
    ]

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_totals", dict(self.support_totals))


def build_reporting_support_state(
    *,
    dm: DesignMatrix,
    groups: list[GroupSlice],
    result: PIRLSResult,
    distribution: Distribution,
    link: Link,
    sample_weight: NDArray,
    y: NDArray,
    offset: NDArray,
    retain_fit_state: bool,
    information_by_group_index: dict[int, NDArray] | None = None,
) -> ReportingSupportState | None:
    """Distill report support from one authoritative terminal fit."""
    structured_indices = [
        index
        for index, matrix in enumerate(dm.group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix | FactorSmoothGroupMatrix)
    ]
    if not structured_indices:
        return None

    full_eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset, link)
    mu = clip_mu(link.inverse(full_eta), distribution)
    variance = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    derivative = link.deriv_inverse(full_eta)
    working_weights = sample_weight * derivative**2 / variance
    supplied = information_by_group_index or {}
    totals: dict[str, StructuredLevelSupport | FactorSmoothLevelSupport] = {}

    for index in structured_indices:
        matrix = dm.group_matrices[index]
        group = groups[index]
        information = supplied.get(index)
        if isinstance(matrix, FactorSmoothGroupMatrix):
            if information is None:
                information, _xtw, _rhs = matrix.factor_smooth_sufficient_stats(
                    working_weights,
                    np.zeros_like(working_weights),
                )
            totals[group.name] = FactorSmoothLevelSupport(
                count=np.bincount(matrix.codes, minlength=matrix.n_levels),
                fit_weight=np.bincount(
                    matrix.codes,
                    weights=sample_weight,
                    minlength=matrix.n_levels,
                ),
                information=information,
            )
            continue

        if not isinstance(matrix, RandomEffectGroupMatrix):  # pragma: no cover
            raise RuntimeError("Structured reporting index has an unsupported matrix.")
        if information is None:
            information = matrix.rmatvec(working_weights)
        unpooled = None
        if not retain_fit_state:
            from superglm.inference.random_effects import (
                vectorized_conditional_unpooled_effect,
            )

            base_eta = full_eta - result.beta[group.sl][matrix.codes]
            unpooled = vectorized_conditional_unpooled_effect(
                codes=matrix.codes,
                n_levels=matrix.n_levels,
                y=y,
                sample_weight=sample_weight,
                base_eta=base_eta,
                distribution=distribution,
                link=link,
                initial=result.beta[group.sl],
            )
        totals[group.name] = StructuredLevelSupport(
            count=np.bincount(matrix.codes, minlength=matrix.n_levels),
            fit_weight=np.bincount(
                matrix.codes,
                weights=sample_weight,
                minlength=matrix.n_levels,
            ),
            information=information,
            unpooled_effect=unpooled,
        )

    return ReportingSupportState(totals)


__all__ = [
    "FactorSmoothLevelSupport",
    "ReportingSupportState",
    "StructuredLevelSupport",
    "build_reporting_support_state",
]
