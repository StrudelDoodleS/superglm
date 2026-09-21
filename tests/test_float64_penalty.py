"""Portable binary64 penalty geometry against independent exact models."""

import numpy as np
import pytest

import superglm.reml.multi_penalty as multi
import superglm.reml.penalty_support as support


@pytest.mark.parametrize("n", [24, 64])
def test_normal_diagonal_geometry_is_certified_in_float64(monkeypatch, n):
    # This exposed the dimension-dependent extended-arithmetic enclosure:
    # normal diagonal geometry was refused when its working type was binary64.
    monkeypatch.setattr(multi, "_LD", np.float64)
    monkeypatch.setattr(support, "_LD", np.float64)
    monkeypatch.setattr(multi, "_U_LD", np.finfo(float).eps / 2)
    monkeypatch.setattr(multi, "_TINY_LD", np.nextafter(0.0, 1.0))
    components = [np.eye(n), 2 * np.eye(n)]
    weights = np.array([2.0, 3.0])
    result = multi.similarity_transform_logdet(components, weights)
    gradient = multi.logdet_s_gradient(result, components, weights)
    hessian = multi.logdet_s_hessian(result, components, weights)
    certificate = result._certificate
    assert result.rank == n
    assert certificate is not None
    eps = np.finfo(float).eps
    assert abs(result.logdet_s_plus - n * np.log(8)) <= certificate.logdet_error + n * eps
    assert np.all(np.abs(gradient - n * np.array([0.25, 0.75])) <= certificate.gradient_error)
    expected_hessian = n * 0.1875 * np.array([[1, -1], [-1, 1]])
    assert np.all(np.abs(hessian - expected_hessian) <= certificate.hessian_error)
    assert np.all(np.abs(result.S_pinv_plus - np.eye(n) / 8) <= certificate.inverse_error)
    np.testing.assert_allclose(gradient, n * np.array([0.25, 0.75]), rtol=32 * eps, atol=0)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=64 * eps, atol=0)
    assert result.S_pinv_plus.dtype == np.float64


@pytest.mark.parametrize("n", [24, 64, 128])
def test_normal_dense_geometry_has_independent_spectral_derivatives(n):
    rng = np.random.default_rng(602 + n)
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
    diagonal = np.linspace(1.0, 2.0, n)
    first = (basis * diagonal) @ basis.T
    second = (basis * diagonal[::-1]) @ basis.T
    weights = np.array([2.0, 3.0])
    result = multi.similarity_transform_logdet([first, second], weights)
    eigenvalues = 2 * diagonal + 3 * diagonal[::-1]
    fractions = 2 * diagonal / eigenvalues
    expected_gradient = np.array([fractions.sum(), n - fractions.sum()])
    cross = np.sum(fractions * (1 - fractions))
    expected_hessian = cross * np.array([[1.0, -1.0], [-1.0, 1.0]])
    # The spectral fixture has condition <= 2. QR, Gram extraction, and the
    # independent eigenvalue formula contribute O(n*eps) backward error.
    allowance = 64 * n * np.finfo(float).eps
    np.testing.assert_allclose(result._gradient, expected_gradient, rtol=allowance, atol=0)
    np.testing.assert_allclose(result._hessian, expected_hessian, rtol=allowance, atol=0)
    assert abs(result.logdet_s_plus - np.log(eigenvalues).sum()) <= allowance * n
    np.testing.assert_allclose(
        result.E_sqrt.T @ result.E_sqrt, 2 * first + 3 * second, rtol=0, atol=allowance * 8
    )


@pytest.mark.parametrize("width", [5, 8])
def test_tensor_difference_penalty_retains_analytic_null_space(width):
    difference = np.diff(np.eye(width), axis=0)
    penalty = difference.T @ difference
    components = [np.kron(penalty, np.eye(width)), np.kron(np.eye(width), penalty)]
    result = multi.similarity_transform_logdet(components, np.array([2.0, 3.0]))
    eigenvalues = 2 - 2 * np.cos(np.arange(width) * np.pi / width)
    total = (2 * eigenvalues[:, None] + 3 * eigenvalues[None, :]).ravel()[1:]
    fractions = (2 * eigenvalues[:, None] * np.ones((1, width))).ravel()[1:] / total
    cross = np.sum(fractions * (1 - fractions))
    assert result.rank == width**2 - 1
    allowance = 64 * width**2 * np.finfo(float).eps
    np.testing.assert_allclose(
        result.Q_zero @ result.Q_zero.T,
        np.ones((width**2, width**2)) / width**2,
        rtol=0,
        atol=allowance,
    )
    np.testing.assert_allclose(
        result._gradient, [fractions.sum(), len(total) - fractions.sum()], rtol=allowance, atol=0
    )
    np.testing.assert_allclose(
        result._hessian, cross * np.array([[1, -1], [-1, 1]]), rtol=allowance, atol=0
    )
    assert abs(result.logdet_s_plus - np.log(total).sum()) <= allowance * len(total)


def test_public_scalar_tensor_reml_completes_at_the_analytic_constant_fit():
    import pandas as pd

    from superglm import Spline, SuperGLM

    first, second = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    frame = pd.DataFrame({"first": first.ravel(), "second": second.ravel()})
    target = np.full(len(frame), 5.0)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={"first": Spline(n_knots=4), "second": Spline(n_knots=4)},
        interactions=[("first", "second")],
    ).fit_reml(frame, target, max_reml_iter=8)
    prediction = model.predict(frame)
    # Constant counts have the exact penalized solution beta=0, intercept=log(5),
    # for every positive smoothing weight. No optimizer reference is needed.
    np.testing.assert_allclose(
        prediction, target, rtol=64 * len(frame) * np.finfo(float).eps, atol=0
    )
    assert prediction.dtype == np.float64


def test_support_does_not_silently_drop_a_column_when_balancing_underflows():
    root = np.diag([1e200, 1e-200])
    with pytest.raises(support.PenaltyNumericalError, match="support|balanc"):
        support._penalty_support_from_roots(
            [root], resolution_limited=[False], input_error_bounds=[np.zeros_like(root)]
        )
