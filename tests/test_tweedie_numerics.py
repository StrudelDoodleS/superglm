from __future__ import annotations

import numpy as np
import pytest

from superglm.distributions import Tweedie
from superglm.profiling.tweedie import estimate_phi


@pytest.mark.parametrize("p", [1.000001, 1.01, 1.5, 1.99, 1.999999])
def test_unit_deviance_is_exactly_zero_when_response_equals_extreme_mean(p: float) -> None:
    values = np.array([1.0e-20, 1.0, 1.0e12])

    actual = Tweedie(p).deviance_unit(values, values)

    np.testing.assert_array_equal(actual, np.zeros_like(values))


def test_pearson_phi_preserves_valid_tiny_means() -> None:
    y = np.array([1.0e-12, 2.0e-12])
    mu = np.array([1.0e-20, 2.0e-20])
    p = 1.5
    expected = float(np.mean((y - mu) ** 2 / mu**p))

    actual = estimate_phi(y, mu, p)

    assert actual == pytest.approx(expected, rel=2.0e-15)
