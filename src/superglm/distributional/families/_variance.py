"""Range recovery for positive variance products."""

import math

from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.kernels.gamma import _binary_product_divide


def _variance_product(numerators: tuple[float, ...], denominators: tuple[float, ...] = ()) -> float:
    """Scale intermediate factors while retaining the public overflow convention."""
    try:
        return _binary_product_divide(numerators, denominators)
    except _NumericalEvaluationError:
        return math.inf
