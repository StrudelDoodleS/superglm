from importlib.util import find_spec

import numpy as np
import pytest
from scipy import special, stats

from ._dirichlet_oracle import mp_row


def kernel():
    assert find_spec("superglm.distributional.kernels.dirichlet") is not None, (
        "Task 5 requires a Dirichlet core kernel"
    )
    from superglm.distributional.kernels.dirichlet import evaluate_dirichlet_rows

    return evaluate_dirichlet_rows


def test_beta_reduction_and_signed_mixed_channels():
    y = np.array([[0.25, 0.75], [0.875, 0.125], [0.5, 0.5]])
    alpha = np.array([[0.2, 4.0], [5.0, 0.3], [20.0, 31.0]])
    mass = np.array([1.0, 2.0, 3.0])
    full = kernel()(y, alpha, mass, derivative_order=2)
    assert full.valid.all()
    np.testing.assert_allclose(
        full.reported_log_likelihood,
        mass * stats.beta.logpdf(y[:, 0], alpha[:, 0], alpha[:, 1]),
        rtol=3e-13,
    )
    for i in range(len(y)):
        value, score, hessian = mp_row(y[i], alpha[i])
        np.testing.assert_allclose(full.reported_log_likelihood[i], mass[i] * value, rtol=3e-13)
        np.testing.assert_allclose(full.score[i], mass[i] * score, rtol=4e-13, atol=1e-14)
        np.testing.assert_allclose(full.hessian_packed[i], mass[i] * hessian, rtol=4e-13)
    assert np.all(full.hessian_packed[:, 1] > 0)
    np.testing.assert_array_equal(full.parameter_independent_carrier, 0)
    for order in (0, 1):
        partial = kernel()(y, alpha, mass, derivative_order=order)
        np.testing.assert_array_equal(partial.reported_log_likelihood, full.reported_log_likelihood)
        assert partial.derivative_order == order


@pytest.mark.parametrize("concentration", [1e2, 1e6, 1e10, 1e12])
@pytest.mark.parametrize("k", [3, 5])
def test_high_concentration_near_mean(concentration, k):
    alpha = concentration * np.arange(1, k + 1, dtype=float) / (k * (k + 1) / 2)
    y = alpha / alpha.sum()
    y[0] = np.nextafter(y[0], 1.0)
    value, score, hessian = mp_row(y, alpha)
    got = kernel()(y[None], alpha[None], np.ones(1), derivative_order=2)
    assert got.valid[0]
    np.testing.assert_allclose(got.reported_log_likelihood[0], value, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(got.score[0] * alpha, score * alpha, rtol=3e-12, atol=3e-13)
    np.testing.assert_allclose(got.hessian_packed[0], hessian, rtol=3e-13)


@pytest.mark.parametrize(
    "alpha,y",
    [
        ([1e-8, 0.2, 40.0], [1e-250, 0.1, 0.9]),
        ([1e-5, 1e10, 3.0], [1e-15, 1 - 3e-10, 3e-10 - 1e-15]),
        ([0.3, 0.4, 0.5], [1e-250, 0.25, 0.75]),
        ([1e12, 2.0, 1e-4], [1 - 2e-12, 1.9999e-12, 1e-16]),
    ],
)
def test_small_shapes_tiny_shares_and_unequal_shapes(alpha, y):
    alpha, y = np.array(alpha), np.array(y)
    want = mp_row(y, alpha)
    got = kernel()(y[None], alpha[None], np.ones(1), derivative_order=2)
    assert got.valid[0]
    for actual, expected in zip(
        (got.reported_log_likelihood[0], got.score[0], got.hessian_packed[0]), want
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)


def test_raw_special_function_mutation_loses_near_mean_score():
    alpha = np.array([1e12, 2e12, 3e12])
    y = alpha / alpha.sum()
    _, score, _ = mp_row(y, alpha)
    raw = special.digamma(alpha.sum()) - special.digamma(alpha) + np.log(y)
    assert np.max(np.abs(alpha * (raw - score))) > 1e-4
    got = kernel()(y[None], alpha[None], np.ones(1), derivative_order=2)
    np.testing.assert_allclose(got.score[0] * alpha, score * alpha, atol=2e-12)


@pytest.mark.parametrize("alpha", [[1e-200, 1.0, 2.0], [1e308, 1e308, 1e308]])
def test_unrepresentable_core_has_finite_placeholders(alpha):
    got = kernel()(np.array([[0.2, 0.3, 0.5]]), np.array([alpha]), np.ones(1), derivative_order=2)
    assert not got.valid[0]
    assert np.isfinite(got.score).all()
    assert np.isfinite(got.hessian_packed).all()


@pytest.mark.parametrize("large", [4.0, 2.0**40])
def test_normalization_and_dominant_share_residual_are_not_rounded_away(large):
    y = np.array([1.0, 2.0**-52, 2.0**-52])
    alpha = large * y
    value, score, hessian = mp_row(y, alpha)
    got = kernel()(y[None], alpha[None], np.ones(1), derivative_order=2)
    assert got.valid[0]
    np.testing.assert_allclose(got.score[0] * alpha, score * alpha, rtol=3e-12)
    np.testing.assert_allclose(got.hessian_packed[0], hessian, rtol=3e-12)
    np.testing.assert_allclose(got.reported_log_likelihood[0], value, rtol=3e-13)


def test_finite_underflowed_natural_hessian_is_not_valid_linked_curvature():
    y = np.array([[1.0, 1e-200, 1e-200]])
    alpha = np.array([[1e200, 1.0, 1.0]])
    result = kernel()(y, alpha, np.ones(1), derivative_order=2)
    # The large component's natural diagonal is ~-2e-400, but its log-link
    # curvature is about 2: finite natural zero must not certify that channel.
    assert not result.valid[0]
    assert "Hessian" in result.failure_reasons[0]
