from __future__ import annotations

import numpy as np

from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    WeightContract,
    resolve_likelihood_weights,
)


def resolved_prior(values: np.ndarray) -> ResolvedLikelihoodWeights:
    """Resolve an explicit prior contract for low-level distributional tests."""
    return resolve_likelihood_weights(
        values,
        n_observations=len(values),
        contract=WeightContract(semantics="prior"),
    )
