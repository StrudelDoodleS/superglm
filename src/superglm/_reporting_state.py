"""Backend-neutral compact support retained for credibility reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import clip_mu
from superglm.group_matrix import FactorSmoothGroupMatrix, RandomEffectGroupMatrix
from superglm.links import stabilize_eta
from superglm.solvers.working_rows import fisher_working_weights

if TYPE_CHECKING:
    from superglm.distributions import Distribution
    from superglm.group_matrix import DesignMatrix
    from superglm.links import Link
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def _frozen(values: NDArray, dtype: type) -> NDArray:
    """A read-only copy of ``values`` as ``dtype``."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StructuredLevelSupport:
    """Compact training support retained for one all-level random effect.

    Its builders form every array per level over the matrix's own codes.
    """

    count: NDArray
    fit_weight: NDArray
    information: NDArray
    unpooled_effect: NDArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", _frozen(self.count, np.int64))
        object.__setattr__(self, "fit_weight", _frozen(self.fit_weight, np.float64))
        object.__setattr__(self, "information", _frozen(self.information, np.float64))
        if self.unpooled_effect is not None:
            object.__setattr__(self, "unpooled_effect", _frozen(self.unpooled_effect, np.float64))


@dataclass(frozen=True)
class FactorSmoothLevelSupport:
    """Compact row support and local Fisher information for a factor smooth.

    Its builders form ``count`` and ``fit_weight`` per level over the matrix's
    own codes, and ``information`` as the symmetrised per-level blocks
    ``(n_levels, block_size, block_size)`` of its sufficient statistics.
    """

    count: NDArray
    fit_weight: NDArray
    information: NDArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", _frozen(self.count, np.int64))
        object.__setattr__(self, "fit_weight", _frozen(self.fit_weight, np.float64))
        object.__setattr__(self, "information", _frozen(self.information, np.float64))


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

    from superglm.solvers.clip_rows import likelihood_information
    from superglm.solvers.mode_score import linear_predictor

    eta_unclipped = linear_predictor(dm, result, offset)
    full_eta = stabilize_eta(eta_unclipped, link)
    mu = clip_mu(link.inverse(full_eta), distribution)
    working_weights = fisher_working_weights(
        distribution=distribution,
        link=link,
        mu=mu,
        eta=full_eta,
        sample_weight=sample_weight,
    )
    # a binomial row the working clip holds reads the likelihood's information
    working_weights = likelihood_information(
        distribution, link, eta_unclipped, full_eta, mu, sample_weight, working_weights
    )
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

        # The filter above admits factor smooths and random effects only.
        matrix = cast(RandomEffectGroupMatrix, matrix)
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
