"""Order-two beta jets, including integer-shape derivative nontermination.

The tolerance is 2e-11 of each order's dimensionless scale (shape partials
rescaled by their coordinates), allowing the fraction's conditioning and
float64 reductions. This is an overlap envelope, not an all-domain certificate.
"""

import importlib

import numpy as np
import pytest

from tests._negative_binomial_window_oracle import beta_jet


@pytest.mark.parametrize(
    "a,b,z",
    [
        (2.0, 1.0, -0.4),
        (0.2, 3.0, -2.0),
        (3.0, 5.0, 0.0),
        (0.01, 0.5, -8.0),
        (8.0, 0.1, 7.0),
        (120.0, 85.0, 0.5),
        (2.0, 2000.0, -9.0),
        (20.0, 0.01, -15.0),
        (0.7, 19.0, 5.0),
    ],
)
@pytest.mark.parametrize("upper", [False, True])
def test_beta_partials_match_independent_integral(a, b, z, upper):
    engine = importlib.import_module("superglm.distributional.kernels._incomplete_beta")
    actual = engine.log_beta_tail_logit_jets([a], [b], [z], upper=upper, order=2)
    expected = beta_jet(a, b, z, upper)
    scales = [np.ones(1), np.array([[a, b, 1.0]]), np.array([[a * a, a * b, a, b * b, b, 1.0]])]
    for got, want, scale in zip(actual, expected, scales, strict=True):
        np.testing.assert_allclose(
            got * scale, want * scale, atol=2e-11 * max(1, np.max(np.abs(want * scale))), rtol=0
        )


def test_fraction_cap_refuses_instead_of_returning_last_iterate(monkeypatch):
    engine = importlib.import_module("superglm.distributional.kernels._incomplete_beta")
    monkeypatch.setattr(engine, "MAX_TERMS", 1)
    jets = engine.log_beta_tail_logit_jets([2.1], [3.2], [-0.4], upper=False, order=2)
    assert all(np.isnan(j).all() for j in jets)


def test_unresolved_small_complement_declines_to_the_count_law_fallback():
    engine = importlib.import_module("superglm.distributional.kernels._incomplete_beta")
    jets = engine.log_beta_tail_logit_jets([1e-8], [50.0], [np.log(1e-9)], upper=True, order=2)
    # The count-law test with size=1e-8, mean=10 and N>=50 independently
    # verifies the fallback; reading this tail from a rounded near-one CDF
    # otherwise gives about 7e-9 relative error in the shape score.
    assert all(np.isnan(j).all() for j in jets)


@pytest.mark.parametrize("order", [True, 1.0, -1, 3])
def test_core_order_is_strict(order):
    engine = importlib.import_module("superglm.distributional.kernels._incomplete_beta")
    with pytest.raises(ValueError, match="order"):
        engine.log_beta_tail_logit_jets([1.0], [2.0], [0.0], upper=False, order=order)
