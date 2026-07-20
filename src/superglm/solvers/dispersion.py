"""Family-specific helpers for estimated Pearson dispersion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Distribution, Tweedie


def dispersion_likelihood_size(
    distribution: Distribution,
    sample_weight: NDArray,
) -> float:
    """Return the likelihood size implied by the family's weight semantics."""
    weights = np.asarray(sample_weight, dtype=np.float64)
    if isinstance(distribution, Tweedie):
        return float(weights.size)
    return float(np.sum(weights, dtype=np.float64))


def pearson_residual_degrees_of_freedom(
    distribution: Distribution,
    sample_weight: NDArray,
    effective_df: float,
) -> float:
    """Return residual d.f. under SuperGLM's family-specific weight contract.

    Non-Tweedie likelihood weights are frequency weights, so their effective
    likelihood size is ``sum(sample_weight)`` and integer weights are exactly
    equivalent to row replication. Tweedie weights are EDM prior weights and
    therefore retain the observation-count correction ``n - edf``.
    """
    likelihood_size = dispersion_likelihood_size(distribution, sample_weight)
    return max(likelihood_size - float(effective_df), 1.0)
